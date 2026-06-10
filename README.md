# ijepa-mini

Mini implementation scaffold for I-JEPA experiments.

The project follows the official I-JEPA organization: the root `main.py` loads
a YAML experiment config, prepares local workers/devices, then dispatches into
the package under `src`.

## Usage

Create and sync the environment with `uv`:

```bash
uv sync
```

Validate the configuration without running training:

```bash
uv run python main.py --dry-run --devices cpu
```

Launch the training entrypoint:

```bash
uv run python main.py train --fname configs/default.yaml --devices cuda:0
```

Other routed tasks are available for later implementation:

```bash
uv run python main.py linear-probe --fname configs/default.yaml --devices cuda:0
uv run python main.py knn --fname configs/default.yaml --devices cuda:0
```
