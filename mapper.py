import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pad_sequence

class Mapper(nn.Module):
    """
    An MLP to map a latent vector z to a sequence of soft prompt vectors.
    """
    def __init__(self, input_dim, output_dim, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim * num_tokens),
        )

    def forward(self, z):
        batch_size = z.shape[0]
        mapped_z = self.mlp(z)
        # Reshape to create a sequence of vectors
        return mapped_z.view(batch_size, self.num_tokens, -1)
    

def get_mapper_training_data(df):
    code_list = df['code'].tolist()
    z_list = df['z'].tolist()
    return code_list, z_list


def collate_fn(batch):
    codes, zs = zip(*batch)
    zs = torch.tensor(np.stack(zs), dtype=torch.float32)
    return list(codes), zs

def sample_prompts_batch_fixed(batch_size):
    """
    Returns a batch of randomly sampled instruction prompts for code generation.
    """
    prompts = [
        "Write a Python function that implements a heuristic algorithm.",
        "Create a heuristic function in Python.",
        "Implement a heuristic algorithm as a Python function.",
        "Generate a Python function for a heuristic solution.",
        "Write a heuristic function in Python to solve the problem.",
        "Develop a Python heuristic algorithm function.",
        "Create a Python function implementing a heuristic approach.",
        "Write a Python heuristic function.",
    ]
    return [random.choice(prompts) for _ in range(batch_size)]

def train_mapper(df, mapper_model, optimizer, decoder_model, decoder_tokenizer, skeleton_prompt,
                 batch_size=4, epochs=20, accumulation_steps=1, max_length=None, verbose=True, val_df=None): 
    """
    Trains the mapper while the decoder is frozen and sharded across GPUs via device_map='auto'.
    """
    # ---------- data ----------
    code_list, z_list = get_mapper_training_data(df)
    dataset = list(zip(code_list, z_list))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Use ReduceLROnPlateau for adaptive learning rate based on training loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # Minimize loss
        factor=0.5,           # Reduce LR by half
        patience=5,           # Wait 5 epochs before reducing
        verbose=verbose,      # Print LR changes
        min_lr=1e-7          # Don't go below this
    )

    mapper_model.train()
    # Freeze decoder parameters but keep it in eval to disable dropout/etc.
    decoder_model.eval()
    for p in decoder_model.parameters():
        p.requires_grad = False
    decoder_model.config.use_cache = False


    # CRITICAL: find the device of the decoder's embedding layer (first shard)
    embed_layer = decoder_model.get_input_embeddings()
    first_dev = embed_layer.weight.device                    # e.g., cuda:0
    embed_dtype = embed_layer.weight.dtype
    mapper_model = mapper_model.to(first_dev)

    pad_id = decoder_tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("decoder_tokenizer.pad_token_id must be set.")
    if not hasattr(decoder_tokenizer, "apply_chat_template") or not decoder_tokenizer.chat_template:
        raise ValueError("decoder_tokenizer.chat_template must be set for instruction formatting.")

    def build_skeleton_texts(batch_size):
        if skeleton_prompt is None:
            if "sample_prompts_batch_fixed" in globals():
                return sample_prompts_batch_fixed(batch_size)
            raise ValueError("skeleton_prompt is required when no prompt sampler is available.")
        if isinstance(skeleton_prompt, str):
            return [skeleton_prompt] * batch_size
        if isinstance(skeleton_prompt, (list, tuple)):
            if len(skeleton_prompt) == batch_size:
                return list(skeleton_prompt)
            if len(skeleton_prompt) > 0:
                return [random.choice(skeleton_prompt) for _ in range(batch_size)]
        raise ValueError("skeleton_prompt must be a string or a list/tuple of strings.")
    

    for epoch in range(epochs):
        total_steps = 0
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        running_ce = 0.0

        for step, (code_batch, z_batch) in enumerate(dataloader):
            # ----- move batch to the decoder's first device -----
            z_batch = z_batch.to(first_dev)

            # ----- build embeddings on first_dev -----
            B = len(code_batch)
            skel_texts = build_skeleton_texts(B)
            code_texts = [f"```python\n{code}\n```" for code in code_batch]

            prompt_id_tensors = []
            target_id_tensors = []
            for skel_text, code_text in zip(skel_texts, code_texts):
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
                target_ids = full_ids[len(prompt_ids):]
                prompt_id_tensors.append(torch.tensor(prompt_ids, dtype=torch.long))
                target_id_tensors.append(torch.tensor(target_ids, dtype=torch.long))

            prompt_ids = pad_sequence(prompt_id_tensors, batch_first=True, padding_value=pad_id).to(first_dev)
            target_ids = pad_sequence(target_id_tensors, batch_first=True, padding_value=pad_id).to(first_dev)
            prompt_embeds = embed_layer(prompt_ids)  # [B, S_prompt_max, D]

            # mapper outputs must match dtype/device of embeddings
            soft_prompt_embeds = mapper_model(z_batch).to(first_dev, dtype=embed_dtype)

            # ensure shapes align: [B, S_soft, D]
            if soft_prompt_embeds.dim() != 3 or soft_prompt_embeds.size(0) != B:
                raise ValueError(f"mapper output must be [B, S_soft, D]; got {tuple(soft_prompt_embeds.shape)}")

            # targets as embeddings for teacher-forcing input
            target_embeds = embed_layer(target_ids)

            # concat: [B, S_skel + S_soft + S_tgt, D]
            inputs_embeds = torch.cat([soft_prompt_embeds, prompt_embeds, target_embeds], dim=1)

            # ----- labels on the same device -----
            ignore_left = torch.full(
                (B, prompt_embeds.size(1) + soft_prompt_embeds.size(1)),
                -100,
                dtype=torch.long,
                device=first_dev,
            )
            tgt_labels = target_ids.masked_fill(target_ids == pad_id, -100)
            labels = torch.cat([ignore_left, tgt_labels], dim=1)  # [B, seq_total]

            left_mask = torch.ones(B, prompt_embeds.size(1) + soft_prompt_embeds.size(1), dtype=torch.long, device=first_dev)
            right_mask = (target_ids!=pad_id).long().to(first_dev)
            att_mask = torch.cat([left_mask, right_mask], dim=1)
            # ----- forward / loss / backward -----
            out = decoder_model(inputs_embeds=inputs_embeds, attention_mask = att_mask, labels=labels)
            loss_ce = out.loss / accumulation_steps

            loss = loss_ce

            loss.backward()
            running += loss.item()
            running_ce += loss_ce.item()


            total_steps += 1
            if total_steps % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Final step if we have accumulated gradients
        if total_steps % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Compute average training loss for this epoch
        avg_train_loss = running / max(1, len(dataloader))
        n = max(1, len(dataloader))

        # Step scheduler with training loss
        scheduler.step(avg_train_loss)

        torch.cuda.empty_cache()
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_train_loss:.4f} | CE: {running_ce/n:.4f}")
        if epoch%5 ==0:
            Mapper_path = "Mapper_Checkpoints/Mapper.pth"
            torch.save(mapper_model.state_dict(), Mapper_path)

    torch.cuda.empty_cache()
    return mapper_model
