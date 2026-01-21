"""
Unified Multi-Task Mapper Training with FSDP

This script mirrors train_unified_mapper.py but wraps the mapper in Fully Sharded
Data Parallel (FSDP) for efficient multi-GPU training on a single node.
"""

import os
from contextlib import nullcontext
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from torch.utils.data import DataLoader, DistributedSampler
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from mapper import Mapper
from train_unified_mapper import (
    TASK_PROMPTS,
    encode_all_heuristics,
    load_all_augmented_heuristics,
    multi_task_collate_fn,
    sample_prompt_with_augmentation,
)


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_distributed() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP training requires CUDA.")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def broadcast_dataframe(df: pd.DataFrame, src: int = 0) -> pd.DataFrame:
    """Broadcast a Pandas DataFrame (with numpy arrays) to all ranks."""
    if not dist.is_initialized():
        return df

    payload = df.to_dict(orient="list") if dist.get_rank() == src else None
    obj_list = [payload]
    dist.broadcast_object_list(obj_list, src=src)
    data = obj_list[0]
    data["z"] = [np.array(z) for z in data["z"]]
    return pd.DataFrame(data)


def save_fsdp_state(fsdp_model: FSDP, path: str):
    """Save a full (gathered) state dict from an FSDP-wrapped module."""
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = fsdp_model.state_dict()
    if is_main_process():
        torch.save(state_dict, path)


def train_unified_mapper_fsdp(
    df: pd.DataFrame,
    mapper_model: FSDP,
    optimizer,
    decoder_model,
    decoder_tokenizer,
    batch_size: int = 4,
    epochs: int = 30,
    accumulation_steps: int = 2,
    max_length: Optional[int] = 2048,
    verbose: bool = True,
    checkpoint_dir: str = "Mapper_Checkpoints_FSDP",
    task_specific_prob: float = 0.60,
    problem_class_prob: float = 0.20,
    general_prob: float = 0.20,
    seed: int = 42,
) -> FSDP:
    """Train mapper with FSDP on a unified multi-task dataset."""
    device = torch.device("cuda", dist.get_rank() if dist.is_initialized() else 0)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    main_process = is_main_process()

    dataset = list(
        zip(
            df["code"].tolist(),
            df["z"].tolist(),
            df["task"].tolist(),
        )
    )

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True, drop_last=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=multi_task_collate_fn,
        pin_memory=True,
    )

    rng = np.random.default_rng(seed + (dist.get_rank() if dist.is_initialized() else 0))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
    )

    mapper_model.train()
    decoder_model.eval()
    for p in decoder_model.parameters():
        p.requires_grad = False
    decoder_model.config.use_cache = False

    embed_layer = decoder_model.get_input_embeddings()
    embed_dtype = embed_layer.weight.dtype
    pad_id = decoder_tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("decoder_tokenizer.pad_token_id must be set.")

    if main_process and verbose:
        print(f"\n{'='*70}")
        print("Training Unified Multi-Task Mapper with FSDP")
        print(f"{'='*70}")
        print(f"  World size:        {world_size}")
        print(f"  Batch size:        {batch_size}")
        print(f"  Accumulation:      {accumulation_steps}")
        print(f"  Epochs:            {epochs}")
        print(f"  Learning rate:     {optimizer.param_groups[0]['lr']}")
        print(f"  Total programs:    {len(dataset)}")
        print(f"  Batches/epoch:     {len(dataloader)}")
        print("Prompt Augmentation:")
        print(f"  Task-specific:     {task_specific_prob*100:.0f}%")
        print(f"  Problem-class:     {problem_class_prob*100:.0f}%")
        print(f"  General:           {general_prob*100:.0f}%\n")

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        running_loss = torch.tensor(0.0, device=device)
        total_steps = 0

        for step, (code_batch, z_batch, task_batch) in enumerate(dataloader):
            z_batch = z_batch.to(device)
            batch_size_actual = len(code_batch)
            code_texts = [f"```python\n{code}\n```" for code in code_batch]

            prompt_id_tensors = []
            target_id_tensors = []

            for task_name, code_text in zip(task_batch, code_texts):
                skel_text = sample_prompt_with_augmentation(
                    task_name=task_name,
                    task_specific_prob=task_specific_prob,
                    problem_class_prob=problem_class_prob,
                    general_prob=general_prob,
                    rng=rng,
                )

                prompt_messages = [{"role": "user", "content": skel_text}]
                full_messages = [
                    {"role": "user", "content": skel_text},
                    {"role": "assistant", "content": code_text},
                ]

                prompt_ids = decoder_tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    truncation=True,
                    max_length=max_length,
                )
                full_ids = decoder_tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=max_length,
                )

                if len(full_ids) < len(prompt_ids):
                    raise ValueError("chat template produced invalid prompt/full lengths.")

                target_ids = full_ids[len(prompt_ids) :]
                prompt_id_tensors.append(torch.tensor(prompt_ids, dtype=torch.long))
                target_id_tensors.append(torch.tensor(target_ids, dtype=torch.long))

            from torch.nn.utils.rnn import pad_sequence

            prompt_ids = (
                pad_sequence(prompt_id_tensors, batch_first=True, padding_value=pad_id).to(device)
            )
            target_ids = (
                pad_sequence(target_id_tensors, batch_first=True, padding_value=pad_id).to(device)
            )

            prompt_embeds = embed_layer(prompt_ids)
            soft_prompt_embeds = mapper_model(z_batch).to(device, dtype=embed_dtype)

            if soft_prompt_embeds.dim() != 3 or soft_prompt_embeds.size(0) != batch_size_actual:
                raise ValueError(
                    f"mapper output must be [B, S_soft, D]; got {tuple(soft_prompt_embeds.shape)}"
                )

            target_embeds = embed_layer(target_ids)
            inputs_embeds = torch.cat([soft_prompt_embeds, prompt_embeds, target_embeds], dim=1)

            ignore_left = torch.full(
                (batch_size_actual, prompt_embeds.size(1) + soft_prompt_embeds.size(1)),
                -100,
                dtype=torch.long,
                device=device,
            )
            tgt_labels = target_ids.masked_fill(target_ids == pad_id, -100)
            labels = torch.cat([ignore_left, tgt_labels], dim=1)

            left_mask = torch.ones(
                batch_size_actual,
                prompt_embeds.size(1) + soft_prompt_embeds.size(1),
                dtype=torch.long,
                device=device,
            )
            right_mask = (target_ids != pad_id).long().to(device)
            att_mask = torch.cat([left_mask, right_mask], dim=1)

            sync_grad = (step + 1) % accumulation_steps == 0
            sync_context = nullcontext() if sync_grad else mapper_model.no_sync()

            with sync_context:
                out = decoder_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=att_mask,
                    labels=labels,
                )
                loss = out.loss / accumulation_steps
                loss.backward()

            running_loss += loss.detach()
            total_steps += 1

            if sync_grad:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if total_steps % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if dist.is_initialized():
            dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        avg_loss = running_loss.item() / world_size / max(1, len(dataloader))

        scheduler.step(avg_loss)
        torch.cuda.empty_cache()

        if main_process and verbose:
            print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"unified_mapper_fsdp_epoch{epoch + 1}.pth"
            )
            save_fsdp_state(mapper_model, checkpoint_path)
            if main_process and verbose:
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    torch.cuda.empty_cache()
    return mapper_model


def main():
    rank = setup_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device("cuda", rank)
    main_process = is_main_process()

    if main_process:
        print(f"\nUsing device: cuda (rank {rank}) | world size: {world_size}\n")

    heuristics_list = None
    unified_df = None

    if main_process:
        heuristics_list = load_all_augmented_heuristics(task_dir="task")
        if len(heuristics_list) == 0:
            raise ValueError("No valid heuristics loaded! Check task directories.")

        print(f"{'='*70}")
        print("Loading Encoder Model")
        print(f"{'='*70}\n")

        encoder_model = SentenceTransformer(
            "BAAI/bge-code-v1",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
        ).to(device)

        encoder_tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-code-v1",
            trust_remote_code=True,
        )

        encoder_model.eval()
        print("✓ Encoder loaded\n")

        unified_df = encode_all_heuristics(
            heuristics_list=heuristics_list,
            encoder_model=encoder_model,
            encoder_tokenizer=encoder_tokenizer,
            device=device,
            batch_size=32,
        )

        del encoder_model
        torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()
    unified_df = broadcast_dataframe(unified_df, src=0)

    if unified_df is None:
        raise RuntimeError("Failed to broadcast dataset to all ranks.")

    if main_process:
        print(f"{'='*70}")
        print("Loading Decoder Model")
        print(f"{'='*70}\n")

    decoder_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map={"": rank},
        trust_remote_code=True,
    )

    decoder_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        trust_remote_code=True,
    )

    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        decoder_tokenizer.pad_token_id = decoder_tokenizer.eos_token_id

    if main_process:
        print("✓ Decoder loaded\n")

    embeddings = np.stack(unified_df["z"].values)
    input_dim = embeddings.shape[1]
    output_dim = decoder_model.config.hidden_size
    num_tokens = 16

    mapper_model = Mapper(
        input_dim=input_dim,
        output_dim=output_dim,
        num_tokens=num_tokens,
    ).to(device)

    mapper_model = FSDP(
        mapper_model,
        device_id=rank,
        sync_module_states=True,
        use_orig_params=True,
    )

    num_params = sum(p.numel() for p in mapper_model.parameters())
    if main_process:
        print(f"Mapper Architecture:")
        print(f"  Input: {input_dim}D (code embeddings)")
        print(f"  Output: {num_tokens} tokens × {output_dim}D (soft prompts)")
        print(f"  Parameters (FSDP-wrapped): {num_params:,}\n")

    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(mapper_model.parameters(), lr=learning_rate)

    checkpoint_dir = "Mapper_Checkpoints_FSDP"
    if main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    trained_mapper = train_unified_mapper_fsdp(
        df=unified_df,
        mapper_model=mapper_model,
        optimizer=optimizer,
        decoder_model=decoder_model,
        decoder_tokenizer=decoder_tokenizer,
        batch_size=4,
        epochs=30,
        accumulation_steps=2,
        max_length=2048,
        verbose=True,
        checkpoint_dir=checkpoint_dir,
    )

    if dist.is_initialized():
        dist.barrier()

    if main_process:
        print(f"\n{'='*70}")
        print("Saving Final Checkpoint")
        print(f"{'='*70}\n")

        task_counts = unified_df["task"].value_counts().to_dict()

        with FSDP.state_dict_type(
            trained_mapper,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            mapper_state = trained_mapper.state_dict()

        checkpoint_data = {
            "model_state_dict": mapper_state,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "num_tokens": num_tokens,
            "tasks_trained": list(TASK_PROMPTS.keys()),
            "total_programs": len(unified_df),
            "programs_per_task": task_counts,
            "epoch": 30,
            "prompt_augmentation": {
                "strategy": "Strategy 1: Task-Specific + Problem-Class + General",
                "task_specific_prob": 0.60,
                "problem_class_prob": 0.20,
                "general_prob": 0.20,
            },
            "fsdp": {
                "world_size": world_size,
                "sharding": "full",
            },
        }

        final_path = os.path.join(checkpoint_dir, "unified_mapper_fsdp.pth")
        torch.save(checkpoint_data, final_path)

        print(f"✓ Final model saved to: {final_path}")
        print(f"\nTraining Complete!")
        print(f"{'='*70}\n")

        print("Training Summary:")
        print(f"  Total programs: {len(unified_df)}")
        print(f"  Tasks trained: {len(set(unified_df['task']))}")
        print(f"  Model parameters: {num_params:,}")
        print("  Prompt augmentation: 60% task-specific, 20% problem-class, 20% general")
        print(f"  Checkpoint: {final_path}")
        print()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
