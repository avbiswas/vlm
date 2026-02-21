from datasets.lm_dataloader import get_dataloader
import torch
import torch.nn as nn
import os
import torch.optim as optim
from tqdm import tqdm
from networks.lm_to_vlm import LM_2_VLM
import numpy as np
from transformers import (
    ViTModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from utils.config_loader import load_config

config = load_config()
c = config["vlm_train"]
paths = config["paths"]

if __name__ == "__main__":
    accelerator = Accelerator(
        gradient_accumulation_steps=c["gradient_accumulation_steps"],
        mixed_precision=c["mixed_precision"], 
        log_with="tensorboard",
        project_dir="logs",
    )

    model_id = "vlm_peft"
    model_name = config["models"]["llm"]

    train_loader, test_loader = get_dataloader(
        batch_size=c["batch_size"], 
        tokenizer_name=model_name,
        device=accelerator.device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id
    
    vit = ViTModel.from_pretrained(config["models"]["vit"]).to(accelerator.device)
    vit.eval()

    qformer_path = os.path.join(paths["models_dir"], "trained_qformer", "best")
    model = LM_2_VLM(
        model_name=model_name,
        qformer_model_path=qformer_path,
        pad_token_id=pad_token_id,
    )

    # --- Optimizer Setup ---
    lr_slow = c["lr_slow"]
    lr_fast = c["lr_fast"]

    qformer_params = model.qformer.get_grouped_params()
    optimizer = optim.AdamW(
        [
            {"params": qformer_params["default"], "lr": lr_slow},
            {"params": qformer_params["cross_blocks"], "lr": lr_slow},
            {"params": qformer_params["query_embeddings"], "lr": lr_slow},
            {"params": model.adapter.parameters(), "lr": lr_fast},
            {
                "params": filter(lambda p: p.requires_grad, model.llm.parameters()),
                "lr": lr_fast,
            },
        ]
    )

    # --- Training Configuration ---
    epochs = c["epochs"]
    total_steps = len(train_loader) * epochs // accelerator.gradient_accumulation_steps

    # --- Cosine LR Scheduler ---
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=c["warmup_steps"], num_training_steps=total_steps
    )

    # --- Prepare with Accelerator ---
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )

    step = 0
    best_test_loss = float("inf")

    def run_inference(model, test_loader, limit_batches=20):
        model.eval()
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                if i >= limit_batches:
                    break

                pixel_values = data["pixel_values"]
                prefix = data["prefix"]
                assistant = data["assistant_prompt"]

                visual_feats = vit(pixel_values).last_hidden_state

                with accelerator.autocast():
                    output = model(visual_feats, prefix, assistant)

                loss = accelerator.gather(output.loss).mean()
                losses.append(loss.item())

        model.train()
        return np.mean(losses) if losses else float("inf")

    model.train()
    accelerator.print(f"Starting training for {epochs} epochs...")
    accelerator.print(f"Total training steps: {total_steps}")

    for epoch in range(epochs):
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for data in pbar:
            with accelerator.accumulate(model):
                pixel_values = data["pixel_values"]
                prefix = data["prefix"]
                assistant = data["assistant_prompt"]

                with torch.no_grad():
                    visual_feats = vit(pixel_values).last_hidden_state

                with accelerator.autocast():
                    output = model(visual_feats, prefix, assistant)
                    loss = output.loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), c["max_grad_norm"])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )

            step += 1

            if step % c["log_every"] == 0 and accelerator.is_local_main_process:
                test_loss = run_inference(model, test_loader)
                accelerator.print(
                    f"Step {step} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss:.4f}"
                )

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    unwrapped_model = accelerator.unwrap_model(model)
                    save_path = os.path.join(paths["models_dir"], model_id, "best")
                    unwrapped_model.save_checkpoint(save_path)
                    accelerator.print(f"✓ New best model saved! Loss: {best_test_loss:.4f}")

            if step % c["save_every"] == 0 and accelerator.is_local_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = os.path.join(paths["models_dir"], model_id, "latest")
                unwrapped_model.save_checkpoint(save_path)

    # Save final model
    if accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = os.path.join(paths["models_dir"], model_id, "final")
        unwrapped_model.save_checkpoint(save_path)
        accelerator.print("Training complete.")