import numpy as np
from networks.q_former import QFormer
import torch
from transformers import DistilBertModel, ViTModel
from datasets.cc_dataloader import get_dataloaders
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from utils.config_loader import load_config, get_config_val

config = load_config()
c = config["q_former_train"]
paths = config["paths"]

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Device: {device}")

bert = DistilBertModel.from_pretrained(config["models"]["qformer_bert"])
vit = ViTModel.from_pretrained(config["models"]["vit"]).to(device)
vit.eval() # ViT is kept frozen during alignment stage

qformer = QFormer(bert)
qformer.to(device)

model_id = "trained_qformer"
lr = c["lr"]
batch_size = c["batch_size"]

train_loader, test_loader = get_dataloaders(
    vit_model=config["models"]["vit"],
    tokenizer=config["models"]["qformer_bert"],
    batch_size=batch_size,
    device=device
)

def calculate_clip_loss(v, t, tau=0.07):
    N = v.size(0)
    v = F.normalize(v, dim=1)
    t = F.normalize(t, dim=1)
    logits = v @ t.t() / tau   # (N, N)
    labels = torch.arange(N, device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels) # rows: image->text
    loss_t2i = F.cross_entropy(logits.t(), labels) # cols: text->image
    loss = 0.5 * (loss_i2t + loss_t2i)
    return loss.mean()

def run_inference(limit_batches=None):
    if limit_batches is None:
        limit_batches = c["limit_eval_batches"]
        
    qformer.eval()
    losses = []
    with torch.no_grad():
        for i, (pixel_values, txt) in enumerate(test_loader):
            if i >= limit_batches:
                break
            
            # Encoding images on GPU
            visual_feats = vit(pixel_values).last_hidden_state
            
            img_emb, txt_emb = qformer(
                visual_feats=visual_feats, 
                text_input_ids=txt["input_ids"],
                text_attention_mask=txt["attention_mask"],
                attention_mode="uni_modal"
            )
            loss = calculate_clip_loss(img_emb, txt_emb, tau=c["tau"])
            losses.append(loss.item())
    qformer.train()

    if not losses:
        return float('inf')
    return np.mean(losses)

grouped_params = qformer.get_grouped_params()
optimizer = optim.Adam(
    [
        {"params": grouped_params["default"], "lr": lr * 0.1},
        {"params": grouped_params["cross_blocks"], "lr": lr},
        {"params": grouped_params["query_embeddings"], "lr": lr},
    ]
)

steps = 0
best_test_loss = np.inf

os.makedirs(paths["models_dir"], exist_ok=True)

for epoch in range(c["epochs"]):
    train_losses = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for (pixel_values, txt) in pbar:
        steps += 1
        
        with torch.no_grad():
            visual_feats = vit(pixel_values).last_hidden_state

        img_emb, txt_emb = qformer(
            visual_feats=visual_feats, 
            text_input_ids=txt["input_ids"],
            text_attention_mask=txt["attention_mask"],
            attention_mode="uni_modal"
        )
        loss = calculate_clip_loss(img_emb, txt_emb, tau=c["tau"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if steps % c["log_every"] == 0:
            tqdm.write(f"Epoch: {epoch+1}, Steps: {steps}, Train loss: {np.mean(train_losses):.4f}")
            train_losses = []

        if steps % c["eval_every"] == 0:
            test_loss = run_inference()
            tqdm.write(f"Steps: {steps}, Test Loss: {test_loss:.4f}")

            if test_loss < best_test_loss:
                best_model_dir = os.path.join(paths["models_dir"], model_id, "best")
                qformer.save_pretrained(best_model_dir)
                tqdm.write(f"✓ New best model saved in {best_model_dir}")
                best_test_loss = test_loss

        if steps % c["save_every"] == 0:
            latest_dir = os.path.join(paths["models_dir"], model_id, "latest")
            qformer.save_pretrained(latest_dir)
            tqdm.write(f"Checkpoint saved in {latest_dir}")