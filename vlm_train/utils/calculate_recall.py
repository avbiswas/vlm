import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import ViTModel

def calculate_recall(model, dataloader, device, vit_model_name, k_values=[1, 5, 10], max_samples=None):
    """
    Calculates Image-to-Text (I2T) and Text-to-Image (T2I) Recall@K.
    """
    model.eval()
    
    vit = ViTModel.from_pretrained(vit_model_name).to(device)
    vit.eval()
    
    image_feats_all = []
    text_feats_all = []
    
    print(f"Extracting features for Recall calculation (max_samples={max_samples})...")
    
    with torch.no_grad():
        count = 0
        for batch in tqdm(dataloader):
            pixel_values, captions = batch
            
            visual_feats = vit(pixel_values.to(device)).last_hidden_state
            
            if isinstance(captions, dict):
                input_ids = captions["input_ids"].to(device)
                attention_mask = captions["attention_mask"].to(device)
            else:
                continue
                
            q_out, t_out = model(
                visual_feats=visual_feats,
                text_input_ids=input_ids,
                text_attention_mask=attention_mask,
                attention_mode="uni_modal"
            )
            
            # Normalize
            img_emb = F.normalize(q_out, dim=1)
            txt_emb = F.normalize(t_out, dim=1)
            
            image_feats_all.append(img_emb.cpu())
            text_feats_all.append(txt_emb.cpu())
            
            count += pixel_values.size(0)
            if max_samples is not None and count >= max_samples:
                break
    
    # Concatenate all features
    image_feats = torch.cat(image_feats_all, dim=0)
    text_feats = torch.cat(text_feats_all, dim=0)
    
    if max_samples is not None:
        image_feats = image_feats[:max_samples]
        text_feats = text_feats[:max_samples]
        
    num_samples = image_feats.size(0)
    print(f"Computing similarity matrix for {num_samples} samples...")
    
    sim_matrix = image_feats @ text_feats.t()
    
    # I2T Recall
    i2t_recall = {k: 0.0 for k in k_values}
    for i in range(num_samples):
        scores = sim_matrix[i]
        topk_indices = scores.topk(max(k_values))[1]
        for k in k_values:
            if i in topk_indices[:k]:
                i2t_recall[k] += 1
    for k in k_values:
        i2t_recall[k] /= num_samples
        
    # T2I Recall
    t2i_recall = {k: 0.0 for k in k_values}
    sim_matrix_t = sim_matrix.t()
    for i in range(num_samples):
        scores = sim_matrix_t[i]
        topk_indices = scores.topk(max(k_values))[1]
        for k in k_values:
            if i in topk_indices[:k]:
                t2i_recall[k] += 1
    for k in k_values:
        t2i_recall[k] /= num_samples
        
    return {
        "i2t": i2t_recall,
        "t2i": t2i_recall,
        "num_samples": num_samples
    }
