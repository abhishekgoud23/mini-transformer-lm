from __future__ import annotations

from src.utils import load_yaml, set_seed
from src.data import DataConfig, build_dataloaders

def main():
    cfg = load_yaml("configs/base.yaml")

    seed = cfg["train"]["seed"]
    set_seed(seed)

    data_cfg = DataConfig(
        name=cfg["dataset"]["name"],
        split_train=cfg["dataset"]["split_train"],
        split_val=cfg["dataset"]["split_val"],
        max_length=cfg["dataset"]["max_length"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=2,
    )

    train_loader, val_loader, tokenizer = build_dataloaders(data_cfg)

    x,y = next(iter(train_loader))
    print("Batch x shape:", tuple(x.shape))
    print("Batch y shape:", tuple(y.shape))
    print("x dtype:", x.dtype, "y dtype:", y.dtype)

    sample_ids = x[0].tolist()
    decoded = tokenizer.decode(sample_ids, skip_special_tokens=True)
    print("\n--- Decoded sample (from x[0]) ---")
    print(decoded[:500])

if __name__ == "__main__":
    main()