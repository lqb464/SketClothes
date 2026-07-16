# Pix2Pix side track

Branched from the training split (`f3c34dd` → shared dataset prep `c322383`).

| Branch | Train |
|--------|-------|
| `main` | SD 1.5 + ControlNet Scribble fine-tune |
| `experiment/pix2pix` | Classic Pix2Pix (U-Net + PatchGAN) |

```bash
cd training
python train_pix2pix.py --output_dir outputs/pix2pix
```
