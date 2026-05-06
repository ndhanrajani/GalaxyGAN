# GalaxyGAN

Conditional WGAN-GP project for generating Galaxy10 DECaLS morphology images.

## Quick Start

1. Create/activate your environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train (example):

```bash
python -m galaxygan.train --epochs 50 --batch-size 64 --out runs/cwgan_gp_local
```

## Minimal Gradio Demo

Launch a small conditional-generation UI from a trained checkpoint:

```bash
python app_gradio.py --checkpoint runs/cwgan_gp_local/checkpoint_epoch0050.pt
```

Then open the URL shown in your terminal (default `http://127.0.0.1:7860`).

### Helpful flags

- `--host 0.0.0.0` to bind externally
- `--port 7861` to change the port
- `--share` to request a Gradio share link

On shared clusters, if Gradio errors on `/tmp`, set writable dirs first, e.g.  
`export TMPDIR=$HOME/.tmp GRADIO_TEMP_DIR=$HOME/.tmp/gradio XDG_CACHE_HOME=$HOME/.cache`.

## Sample figures (in repo)

Curated PNGs for write-ups and grading: see **[`docs/samples/README.md`](docs/samples/README.md)**.