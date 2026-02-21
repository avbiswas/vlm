from typing import List, Optional, Tuple, Dict, Any
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader, random_split
import torch
from transformers import AutoTokenizer
from .base_dataset import CCBaseDataset

class CCImageCaptionDataset(CCBaseDataset):
    """
    Torch-style Dataset for Q-Former training (Stage 1).
    Returns: (preprocessed_image_tensor, caption_string)
    """
    def __init__(
        self,
        dataset_root: str = "dataset",
        vit_model_name: str = "google/vit-base-patch16-224",
        tokenizer_name: Optional[str] = None,
    ) -> None:
        super().__init__(dataset_root, vit_model_name)
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        ex = self._examples[idx]
        
        with Image.open(ex.image_path) as im:
            image = im.convert("RGB").copy()

        # Preprocess image but don't run through ViT here to allow batching on GPU
        pixel_values = self.vit_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        return pixel_values, ex.caption

def collate_fn(
    batch: List[Tuple[torch.Tensor, str]], 
    tokenizer: Optional[AutoTokenizer] = None,
    device: str = "cpu"
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    images, captions = zip(*batch)
    image_tensors = torch.stack(images, dim=0).to(device)
    
    if tokenizer is not None:
        tokenized = tokenizer(
            list(captions),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        return image_tensors, tokenized
    else:
        return image_tensors, list(captions)

def get_dataloaders(
    vit_model="google/vit-base-patch16-224",
    tokenizer="distilbert/distilbert-base-uncased",
    batch_size=16,
    split_ratio=0.9,
    seed=42,
    device="cpu"
):
    dataset = CCImageCaptionDataset(vit_model_name=vit_model, tokenizer_name=tokenizer)

    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    collate_fn_with_args = partial(collate_fn, tokenizer=dataset.tokenizer, device=device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0, # Keep 0 for stability in some envs, can be increased by user
        collate_fn=collate_fn_with_args,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_with_args,
    )

    return train_loader, test_loader