from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import yaml

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)