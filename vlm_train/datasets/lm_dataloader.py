from typing import Any, Dict, List, Optional, Tuple
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader, random_split
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import random
from .base_dataset import CCBaseDataset

class LMDataset(CCBaseDataset):
    """
    Torch-style Dataset for VLM fine-tuning (Stage 2).
    """
    def __init__(
        self,
        dataset_root: str = "dataset",
        vit_model_name: str = "google/vit-base-patch16-224",
        tokenizer_name: str = "HuggingFaceTB/SmolLM-135M-Instruct',
    ) -> None:
        super().__init__(dataset_root, vit_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.prompts = [
            "Tell me about this image:",
            "Describe this picture.",
            "What do you see in this image?",
            "Provide a description of the photo.",
            "Can you explain what is shown in this image?",
            "What is in this picture?",
            "Describe the contents of this image.",
            "Give me a summary of what's shown here.",
            "What can you see here?",
            "Explain the visual content of this image.",
            "Describe this image in detail.",
            "What's happening in this photo?",
        ]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self._examples[idx]
        
        with Image.open(ex.image_path) as im:
            image = im.convert("RGB").copy()

        pixel_values = self.vit_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        random_prompt = random.choice(self.prompts)
        
'''
We don't apply_chat_template here because it makes padding more complex in collator for dynamic prefixes if we want to be efficient.
but to keep it simple and compatible with existing Stage 2 logic:
'''
        user_prompt_ids = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Answer the user's question truthfully"},
                {"role": "user", "content": random_prompt},
            ],
            return_tensors="pt",
        ).squeeze(0)

        assistant_prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": ex.caption}],
            return_tensors="pt",
            add_generation_prompt=False,
        ).squeeze(0)

        # Truncate after EOS if present
        eos_positions = (assistant_prompt_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            last_eos_idx = eos_positions[-1].item()
            assistant_prompt_ids = assistant_prompt_ids[: last_eos_idx + 1]

        return {
            "pixel_values": pixel_values,
            "prefix": user_prompt_ids,
            "assistant_prompt": assistant_prompt_ids,
        }

class LMCollator:
    def __init__(self, tokenizer, device="cpu"):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([item["pixel_values"] for item in batch]).to(self.device)
        prefixes = [item["prefix"] for item in batch]
        assistant_prompts = [item["assistant_prompt"] for item in batch]

        pad_id = self.tokenizer.pad_token_id
        
        max_prefix_len = max([p.size(0) for p in prefixes])
        prefixes_padded = torch.full((len(prefixes), max_prefix_len), pad_id, dtype=torch.long)
        for i, p in enumerate(prefixes):
            prefixes_padded[i, -len(p) :] = p

        assistant_prompts_padded = pad_sequence(assistant_prompts, batch_first=True, padding_value=pad_id)

        return {
            "pixel_values": pixel_values,
            "prefix": prefixes_padded.to(self.device),
            "assistant_prompt": assistant_prompts_padded.to(self.device),
        }

def get_dataloader(
    batch_size=4, 
    split_ratio=0.9, 
    seed=42, 
    tokenizer_name="HuggingFaceTB/SmolLM-135M-Instruct",
    device="cpu"
):
    dataset = LMDataset(tokenizer_name=tokenizer_name)
    
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    collator = LMCollator(dataset.tokenizer, device=device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    return train_loader, test_loader