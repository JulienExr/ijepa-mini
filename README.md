# ijepa-mini

Mini implementation scaffold for local I-JEPA experiments.

The project follows the official I-JEPA organization: the root `main.py` loads
a YAML experiment config, prepares local workers/devices, then dispatches into
the package under `src`.

## Usage

Create and sync the environment with `uv`:

```bash
uv sync
```

Validate the default ImageNet50-200 configuration without running training:

```bash
uv run python main.py --dry-run --devices cpu
```

Prepare the ImageNet50-200 subset from an ImageNet-style source tree:

```bash
uv run python scripts/prepare_imagenet50_subset.py \
  --source /path/to/imagenet \
  --destination data/imagenet50-200
```

Launch the current small I-JEPA pretraining protocol:

```bash
uv run python main.py train \
  --fname configs/imagenet50_200_vit_small_original_mask_jepa.yaml \
  --devices cuda:0
```

Run the current comparison protocol:

```bash
uv run python scripts/compare_imagenet50_linear_probe.py \
  --data-root data/imagenet50-200 \
  --jepa-config configs/imagenet50_200_vit_small_original_mask_jepa.yaml \
  --jepa-checkpoint outputs/imagenet50-200-vit-small-original-mask-jepa/checkpoints/imagenet50-200-vit-small-original-mask-jepa_latest.pt \
  --output-dir outputs/imagenet50-200-vit-small-original-mask-comparison \
  --device cuda:0
```
