from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pyarrow.parquet as pq
import os
from torch.utils.data import Dataset
from transformers import ViTImageProcessor, ViTModel

@dataclass(frozen=True)
class CCExample:
    image_path: Path
    caption: str

class CCBaseDataset(Dataset):
    """
    Base Dataset for Conceptual Captions images.
    Consolidates shared logic between Stage 1 and Stage 2 dataloaders.
    """
    def __init__(
        self,
        dataset_root: str | Path = "dataset",
        vit_model_name: str = "google/vit-base-patch16-224",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.images_root = self.dataset_root / "cc_images"
        self.index_parquet = self.dataset_root / "conceptual-captions-200k.parquet"

        self.vit_processor = ViTImageProcessor.from_pretrained(vit_model_name)
        self.vit_model = ViTModel.from_pretrained(vit_model_name)
        
        self._examples: List[CCExample] = self._build_index()

    def _build_image_paths(self) -> Dict[int, str]:
        """Scans the image directory and builds a mapping from index to path."""
        jpg_files = {}
        if not self.images_root.exists():
            return jpg_files
            
        for subdir in self.images_root.iterdir():
            if not subdir.is_dir():
                continue
            for file in subdir.iterdir():
                if file.is_file() and file.suffix.lower() == ".jpg":
                    if file.name.startswith("."):
                        continue
                    try:
                        file_idx = int(file.name.split(".")[0])
                        jpg_files[file_idx] = str(file)
                    except ValueError:
                        continue
        return jpg_files

    def _build_index(self) -> List[CCExample]:
        """Cross-references images on disk with the metadata parquet file."""
        if not self.index_parquet.exists():
            print(f"Warning: Index parquet not found at {self.index_parquet}")
            return []

        image_files = self._build_image_paths()
        table = pq.read_table(self.index_parquet, columns=["caption"])
        captions = table["caption"].to_pylist()
        
        out: List[CCExample] = []
        for idx, caption in enumerate(captions):
            if idx in image_files:
                out.append(
                    CCExample(
                        image_path=Path(image_files[idx]),
                        caption=caption or "",
                    )
                )
        return out

    def __len__(self) -> int:
        return len(self._examples)