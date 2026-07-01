import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_fidelity import calculate_metrics


class NpzImageDataset(Dataset):
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        if self.path.suffix != ".npz":
            raise ValueError(f"Expected .npz input, got: {self.path}")

        obj = np.load(self.path)
        if "arr_0" not in obj.files:
            raise KeyError(f"{self.path} has no arr_0. Available keys: {obj.files}")
        self.arr = obj["arr_0"]
        if self.arr.ndim != 4:
            raise ValueError(f"Expected 4D image array, got shape {self.arr.shape}")

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        img = self.arr[idx]
        if img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            if img.max() <= 1.5:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        return torch.from_numpy(img.copy())


def main():
    parser = argparse.ArgumentParser(
        description="Compute IS/FID/Precision/Recall for generated NPZ images with torch-fidelity."
    )
    parser.add_argument("--ref", required=True, help="Reference ImageNet batch .npz with arr_0.")
    parser.add_argument("--sample", required=True, help="Generated sample .npz with arr_0.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    ref_ds = NpzImageDataset(args.ref)
    sample_ds = NpzImageDataset(args.sample)

    print(f"[INFO] ref: {args.ref} ({len(ref_ds)} images)")
    print(f"[INFO] sample: {args.sample} ({len(sample_ds)} images)")
    print(f"[INFO] device: {args.device}")

    metrics = calculate_metrics(
        input1=sample_ds,
        input2=ref_ds,
        cuda=(args.device == "cuda"),
        batch_size=args.batch_size,
        isc=True,
        fid=True,
        prc=True,
        verbose=True,
    )

    print("[RESULT] Inception Score:", metrics.get("inception_score_mean"))
    print("[RESULT] Inception Score Std:", metrics.get("inception_score_std"))
    print("[RESULT] FID:", metrics.get("frechet_inception_distance"))
    print("[RESULT] Precision:", metrics.get("precision"))
    print("[RESULT] Recall:", metrics.get("recall"))


if __name__ == "__main__":
    main()
