import json
import shutil
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
CHECKPOINT_DIR = REPO_ROOT / "models"
REQUIRED_FILES = ("dataset.pt", "vocab.json")

DATASET_PROJECT = "Chess_LSTM"
DATASET_NAME = "Chess UCI Dataset"


def ensure_data() -> None:
    """Если данных нет локально качаем их из ClearML Dataset."""
    missing = [name for name in REQUIRED_FILES if not (DATA_DIR / name).exists()]
    if not missing:
        print(f"[data] файл: {DATA_DIR}")
        return
    from clearml import Dataset
    print(f"[data] {missing}")
    dataset = Dataset.get(dataset_project=DATASET_PROJECT, dataset_name=DATASET_NAME)
    print(f"[data] взят dataset id: {dataset.id}")
    cache = Path(dataset.get_local_copy())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in missing:
        source = next(cache.rglob(name))
        shutil.copy(source, DATA_DIR / name)
        print(f"[data]   скопирован {name}")


def load_vocab() -> dict:
    with open(DATA_DIR / "vocab.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_dataset() -> dict:
    return torch.load(DATA_DIR / "dataset.pt", weights_only=False)