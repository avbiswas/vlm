import torch
import torch.nn.functional as F
from transformers import ViTModel, AutoTokenizer
from networks.q_former import QFormer
from networks.lm_to_vlm import LM_2_VLM
from datasets.cc_dataloader import get_dataloaders
from utils.calculate_recall import calculate_recall
from utils.utils import create_similarity_grid
from utils.config_loader import load_config
import os
from PIL import Image

def run_retrieval_eval(qformer, vit, test_loader, device, output_dir):
    """Computes Recall@K metrics and generates a similarity grid."""
    print("\n--- Starting Retrieval Evaluation ---")
    metrics = calculate_recall(qformer, test_loader, device, k_values=[1, 5, 10], max_samples=20)
    
    samples = []
    scores_list = []
    
    qformer.eval()
    with torch.no_grad():
        for i, (pixel_values, txt) in enumerate(test_loader):
            if i >= 1: # Just take first batch for grid
                break
            
            visual_feats = vit(pixel_values).last_hidden_state
            q_out, t_out = qformer(
                visual_feats=visual_feats,
                text_input_ids=txt["input_ids"],
                text_attention_mask=txt["attention_mask"],
                attention_mode="uni_modal"
            )
            
            img_emb = F.normalize(q_out, dim=1)
            txt_emb = F.normalize(t_out, dim=1)
            
            scores = img_emb @ txt_emb.t()
            
            # This is a bit tricky since dataloader returns tensors, we'll just take the first N samples
            for j in range(min(8, pixel_values.size(0))):
                # We need to get the original image somehow or just show the tensor
                pass

    print(f"I2T Recall: {metrics['i2t']}")
    print(f"T2I Recall: {metrics['t2i']}")

def run_generation_eval(vlm, vit, tokenizer, test_loader, device):
    vlm.eval()
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i >= 5: # Test on 5 samples
                break
            
            pixel_values = data["pixel_values"]
            prefix = data["prefix"]
            
            visual_feats = vit(pixel_values).last_hidden_state
            
            output_ids = vlm.generate(
                img=visual_feats, 
                prefix_ids=prefix,
                max_new_tokens=50
            )
            
            captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for j, cap in enumerate(captions):
                print(f"Sample {i*len(captions)+j}: {cap}")

if __name__ == "__main__":
    config = load_config()
    paths = config["paths"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vit = ViTModel.from_pretrained(config["models"]["vit"]).to(device)
    vit.eval()
    
    qformer_path = os.path.join(paths["models_dir"], "trained_qformer", "best")
    if os.path.exists(qformer_path):
        qformer = QFormer.from_pretrained(qformer_path).to(device)
        _, test_loader_q = get_dataloaders(batch_size=8, device=device)

    vlm_path = os.path.join(paths["models_dir"], "vlm_peft", "best")
    if os.path.exists(vlm_path):
        tokenizer = AutoTokenizer.from_pretrained(config["models"]["llm"])
        vlm = LM_2_VLM(model_name=config["models"]["llm"], qformer_model_path=qformer_path)
        vlm.load_checkpoint(vlm_path)
        vlm.to(device)
        
        from datasets.lm_dataloader import get_dataloader
        _, test_loader_lm = get_dataloader(batch_size=1, device=device)
        run_generation_eval(vlm, vit, tokenizer, test_loader_lm, device)