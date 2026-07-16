"""
Side-track Pix2Pix trainer (sketch → garment photo).

Reuses the same local photos/sketches layout as ControlNet training.
Adapted from the old project: https://github.com/lqb464/Sketch-to-Image-by-Pix2Pix

Usage:
  cd training
  python train_pix2pix.py --output_dir outputs/pix2pix
  python train_pix2pix.py --max_train_samples 200 --epochs 5   # smoke test
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

# Allow `from utils.dataset import load_hf_dataset` when run from training/pix2pix/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pix2pix_networks import (  # noqa: E402
    GANLoss,
    NLayerDiscriminator,
    UnetGenerator,
    init_weights,
)
from utils.dataset import load_hf_dataset, split_train_val  # noqa: E402


class Pix2PixPairDataset(Dataset):
    """A = sketch (input), B = photo (target). Both in [-1, 1]."""

    def __init__(
        self,
        hf_dataset,
        resolution: int = 256,
        random_flip: bool = True,
    ) -> None:
        self.data = hf_dataset
        self.resolution = resolution
        self.random_flip = random_flip
        self.resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.data[index]
        photo = self.resize(row["image"].convert("RGB"))
        sketch = self.resize(row["sketch"].convert("RGB"))

        if self.random_flip and random.random() < 0.5:
            photo = transforms.functional.hflip(photo)
            sketch = transforms.functional.hflip(sketch)

        return {
            "A": self.to_tensor(sketch),
            "B": self.to_tensor(photo),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Pix2Pix sketch→photo (side track)")
    p.add_argument("--output_dir", type=str, default="outputs/pix2pix")
    p.add_argument("--dataset_id", type=str, default="")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_l1", type=float, default=100.0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--sample_every", type=int, default=1)
    return p.parse_args()


@torch.no_grad()
def save_validation_grid(
    net_g: nn.Module,
    val_loader: DataLoader,
    out_path: Path,
    device: torch.device,
    n: int = 4,
) -> None:
    net_g.eval()
    batch = next(iter(val_loader))
    a = batch["A"][:n].to(device)
    b = batch["B"][:n].to(device)
    fake = net_g(a)
    # [-1,1] → [0,1]
    grid = torch.cat([a, fake, b], dim=0)
    grid = (grid + 1.0) * 0.5
    save_image(grid, out_path, nrow=n)
    net_g.train()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    samples_dir = out_dir / "samples"
    ckpt_dir = out_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hf = load_hf_dataset(args.dataset_id, "train", args.max_train_samples)
    train_hf, val_hf = split_train_val(hf, val_ratio=0.02, seed=args.seed)
    print(f"Pix2Pix dataset: {len(train_hf)} train / {len(val_hf)} val @ {args.resolution}px")

    train_ds = Pix2PixPairDataset(train_hf, resolution=args.resolution, random_flip=True)
    val_ds = Pix2PixPairDataset(val_hf, resolution=args.resolution, random_flip=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=max(1, min(4, len(val_ds))), shuffle=False)

    net_g = init_weights(UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=64, use_dropout=True)).to(device)
    net_d = init_weights(NLayerDiscriminator(input_nc=6, ndf=64, n_layers=3)).to(device)

    criterion_gan = GANLoss().to(device)
    criterion_l1 = nn.L1Loss()
    opt_g = torch.optim.Adam(net_g.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=args.lr, betas=(0.5, 0.999))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Pix2Pix epoch {epoch}/{args.epochs}")
        for batch in pbar:
            real_a = batch["A"].to(device)
            real_b = batch["B"].to(device)

            # --- Discriminator ---
            opt_d.zero_grad(set_to_none=True)
            fake_b = net_g(real_a)
            pred_fake = net_d(torch.cat([real_a, fake_b.detach()], dim=1))
            loss_d_fake = criterion_gan(pred_fake, False)
            pred_real = net_d(torch.cat([real_a, real_b], dim=1))
            loss_d_real = criterion_gan(pred_real, True)
            loss_d = 0.5 * (loss_d_fake + loss_d_real)
            loss_d.backward()
            opt_d.step()

            # --- Generator ---
            opt_g.zero_grad(set_to_none=True)
            pred_fake_for_g = net_d(torch.cat([real_a, fake_b], dim=1))
            loss_g_gan = criterion_gan(pred_fake_for_g, True)
            loss_g_l1 = criterion_l1(fake_b, real_b) * args.lambda_l1
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            opt_g.step()

            global_step += 1
            pbar.set_postfix(D=f"{loss_d.item():.3f}", G=f"{loss_g.item():.3f}", L1=f"{loss_g_l1.item():.3f}")

        if epoch % args.sample_every == 0:
            save_validation_grid(net_g, val_loader, samples_dir / f"epoch-{epoch:04d}.png", device)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {"net_g": net_g.state_dict(), "net_d": net_d.state_dict(), "epoch": epoch},
                ckpt_dir / f"epoch-{epoch:04d}.pt",
            )
            torch.save(net_g.state_dict(), ckpt_dir / "latest_net_G.pth")
            print(f"Saved checkpoint → {ckpt_dir}")

    print(f"Done. Samples: {samples_dir}")


if __name__ == "__main__":
    main()
