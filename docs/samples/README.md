# Sample outputs (for reports / GitHub)

Small curated images committed to the repo. Full checkpoints and all epochs stay on cluster under `runs/` (gitignored).

## Conditional WGAN-GP ablation (`N_CRITIC=3`, `LR=1e-4`, 200 epochs)

One row of generated samples per epoch (10 morphology classes).

| Epoch | File |
|------:|------|
| 1 | [`nc3_lr1e4_e200_epoch0001.png`](nc3_lr1e4_e200_epoch0001.png) |
| 50 | [`nc3_lr1e4_e200_epoch0050.png`](nc3_lr1e4_e200_epoch0050.png) |
| 100 | [`nc3_lr1e4_e200_epoch0100.png`](nc3_lr1e4_e200_epoch0100.png) |
| 150 | [`nc3_lr1e4_e200_epoch0150.png`](nc3_lr1e4_e200_epoch0150.png) |
| 200 | [`nc3_lr1e4_e200_epoch0200.png`](nc3_lr1e4_e200_epoch0200.png) |

## Longer baseline runs (comparison)

| Run | File |
|-----|------|
| `cwgan_total300`, epoch 300 | [`baseline_total300_epoch0300.png`](baseline_total300_epoch0300.png) |
| `cwgan_total1200`, epoch 1200 | [`baseline_total1200_epoch1200.png`](baseline_total1200_epoch1200.png) |

## Gradio demo (conditional generation UI)

Screenshots from local browser via SSH port forwarding to Quest.

| | File |
|-|------|
| 1 | [`gradio_demo_01.png`](gradio_demo_01.png) |
| 2 | [`gradio_demo_02.png`](gradio_demo_02.png) |
| 3 | [`gradio_demo_03.png`](gradio_demo_03.png) |
| 4 | [`gradio_demo_04.png`](gradio_demo_04.png) |
