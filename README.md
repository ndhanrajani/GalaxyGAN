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

## GalaxyGAN.pdf (verbatim text)

GalaxyGAN: Conditional Generation of Galaxy
Morphologies with WGAN-GP
Natasha Dhanrajani
GitHub URL
https://github.com/ndhanrajani/GalaxyGAN
Project Overview
This project explores class-conditional image generation using a Wasserstein Generative Adversarial Net-
work with Gradient Penalty (WGAN-GP) model trained on data comprising images of galaxies. Specifi-
cally, this project aims to design a class-conditional generator model capable of producing galaxy images
with the desired morphology class labels while achieving consistent model training on Northwestern Quest
infrastructure.
Galaxy10 DECaLS dataset by AstroNN is used. The dataset comprises ˜17,700 images of galaxies
belonging to 10 different morphology classes: mergers, smooth ellipticals, spirals, and edge-on galaxies.
It makes sense to use the dataset in this project due to the distinct appearance of each class.
A generator architecture based on DCGAN was chosen; class conditioning is performed using embed-
dings of the morphology class label. The inputs to the generator comprise:
• 128-dimensional latent vector
• Learned embeddings of the galaxy morphology class label
Inputs to the discriminator (critic) besides those above include embeddings of galaxy morphology
class label distributed throughout the image grid and concatenated with the input image tensor.
The output of the generator is a 69 × 69 RGB image. To accommodate the non-square target size,
the generator produces an output of size 64 × 64 which is then resized to the necessary dimensions.
All models and training processes were implemented in PyTorch using Slurm batch jobs on North-
western Quest GPU nodes.
Extra Criteria Pursued
1. Conditional Generation by Morphology (Completed)
Conditioning on the morphology label was done using embedding of the class label in the generator and
discriminator. Sample grids were created for all 10 morphological classes with outputs varying based
on the conditioning label. The generated images contained galaxy-like structures, such as bright cores,
elongated edge-on structures, diffuse merger-like objects, and smooth elliptical.
2. Interactive GUI via Gradio (Completed)
A Gradio-based interface was built for interactive use of the trained GAN allowing users to select a galaxy
morphology class, provide a seed value, and generate a corresponding galaxy image from a specific GAN
checkpoint.
1

-- 1 of 3 --

3. Training Evaluation and Analysis (Partially Completed)
Qualitative analysis of the training procedure using sample grids and model checkpoints saved throughout
the process was provided. Model outputs of curated samples were analyzed at epochs
1, 50, 100, 150, 200, 300, 1200
The following observations were made:
• Structure learning occurred in earlier epochs.
• Peak performance was recorded in intermediate epochs.
• Degradation began occurring during later stages.
Although a more quantitative approach to evaluation involving FID and Inception Score was planned
initially, it could not be completed due to lack of time.
Difficulties Faced and Solutions
1. CUDA and PyTorch Compatibility Problems
One of the main issues with running the code on Northwestern Quest was compatibility of PyTorch
CUDA. The initial code was executed using a wrong PyTorch CUDA package and exhibited issues like
CUDA initialization errors, silent fallback to CPU mode, and very slow training. The problem was solved
by reinstalling PyTorch with the correct CUDA version and implementing a fail-fast approach to stop
jobs that run into CUDA problems.
2. Training Speed
At the beginning, the training was extremely slow with multiple hours needed per epoch. Some of the
reasons for that were:
• Multiple read-throughs of data stored in common HDF5 file.
• Excessively expensive resizing operation performed by the data loader.
• Data loader parameter num workers = 0.
Several approaches were taken to mitigate these issues:
• Pre-loading images into memory.
• Copying dataset onto a local node disk drive.
• Applying multiprocessing optimizations to the DataLoader.
• Enabling persistent workers.
3. GAN Instability and Collapse
The first stage of training revealed classic signs of GAN training instability, such as checkerboard effect,
noise, and low diversity in the output. Peak performance was reached around epochs 100–150 after which
the output started to degrade. This issue was resolved by adjusting hyperparameters, namely lowering
learning rate and reducing the number of critic updates per iteration (n critic = 3). The process was
monitored using qualitative evaluation of the output at each epoch. This helped determine an early
stopping criteria.
2

-- 2 of 3 --

Results
The trained generator produces images resembling galaxy structures without exhibiting any constant
noise patterns. Some characteristics of images produced include:
• Bright central core
• Diffuse light distribution
• Elongated, edge-on galaxy shape
• Characteristics associated with mergers.
The best results are attained with epochs 100–150 using:
n critic = 3, learning rate = 1 × 10−4
Although the images are low-resolution and somewhat noisy, the project successfully demonstrates
conditional image generation and meaningful structure learning on galaxy morphology data.
Future Work
Some directions to pursue in the future include:
• Substitution of transposed convolutions for up-sampling + convolution operations to alleviate the
checkerboard effect.
• Using more quantitative criteria, e.g. FID and Inception Score.
• Performing analysis of latent spaces using techniques like interpolation, PCA, t-SNE.
3

-- 3 of 3 --