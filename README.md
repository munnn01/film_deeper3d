# Video Swin Lite preprocessing with a FiLM deeper-3D codec proxy

Task-aware video preprocessing with the requested pipeline kept intact:

```text
video -> Video Swin Lite -> H.264/H.265 -> reconstruction
      -> frozen analyzer -> task
```

During training, a frozen differentiable proxy runs beside the real codec:

```text
                            +-> frozen codec proxy -- backward gradients --+
                            |                                               |
video -> trainable preprocessor -> real H.264/H.265 -> reconstruction ------+
                                                    -> frozen analyzer -> task
```

The real FFmpeg codec always determines reconstruction and measured BPP in the
forward pass. The proxy supplies only the backward Jacobian:

```python
reconstruction = proxy_reconstruction + (
    real_reconstruction - proxy_reconstruction
).detach()

bpp = proxy_bpp + (real_bpp - proxy_bpp).detach()
```

Consequently, task and rate-distortion loss values correspond to the standard
codec while gradients still reach only the preprocessor. Codec proxy and analyzer
parameters remain frozen.

## FiLM deeper-3D proxy

The differentiable proxy is a compact 3-D residual encoder-decoder rather than
the earlier two-layer Conv3D model:

```text
RGB clip + per-sample QP
  -> spatial /2 Conv3D -> FiLM -> residual 3-D blocks ------------ skip ----+
  -> spatial /4 Conv3D -> FiLM -> residual 3-D blocks ------ skip ----+      |
  -> spatial /8 Conv3D -> FiLM -> residual 3-D bottleneck           |      |
  -> softened STE quantization                                      |      |
  -> up /4 + skip -> FiLM -> residual 3-D blocks <------------------+      |
  -> up /2 + skip -> FiLM -> residual 3-D blocks <-------------------------+
  -> RGB residual -> input + delta -> proxy reconstruction
```

Every FiLM layer applies `gamma(QP) * feature + beta(QP)`, so mixed-QP samples
can change feature scale as well as bias. The three spatial scales and residual
blocks give a receptive field larger than a 64x64 H.264 coding-tree region.
The rate head uses mean magnitude, spatial variance, smooth sparsity and temporal
change statistics. The proxy remains frozen during preprocessor training, but
autograd still differentiates its output with respect to the preprocessed clip.

## Video Swin Lite preprocessor

`VideoSwinLitePreprocessor` is a compact dense video transformer:

```text
BTCHW RGB video
  -> Conv3D spatial patch embedding, patch=(1,4,4), 3 -> 48 channels
  -> depthwise Conv3D positional encoding
  -> four alternating regular/shifted 3-D Swin blocks
       window=(4,8,8), heads=4, MLP ratio=4
  -> LayerNorm
  -> ConvTranspose3D spatial reconstruction, 48 -> RGB
  -> tanh * 0.25
  -> input + RGB residual
```

Temporal resolution is never downsampled. Shifted windows exchange information
between neighboring clips and spatial regions while avoiding global space-time
attention. The RGB head is zero-initialized, so a new model is exactly the identity
mapping. The earlier factorized ViT and CNN remain available as `--preprocessor vit`
and `--preprocessor cnn` for ablation; `swin` is the default.

## Objective

```text
L = alpha * (L_D + lambda * L_R) + L_Acc
```

- `L_D`: MSE between source video and real codec reconstruction.
- `L_R`: measured elementary-stream BPP from H.264/H.265.
- `L_Acc`: cross-entropy from the frozen Kinetics-400 analyzer.
- Defaults: `alpha=10`, `lambda=0.001`, Adam `lr=1e-4`.

## Requirements

```bash
pip install -r requirements.txt
```

FFmpeg must include `libx264` and/or `libx265`. There is one unified requirements
file; no Kaggle-specific requirements file is needed.

## Training order

First build a deterministic codec cache. The raw pipe is checked against the
legacy PNG path before caching, two FFmpeg workers run concurrently, and each
source clip is stored only once as `uint8`. `train/` and `val/` have separate
cache trees; when the dataset has no `val/`, the split is stratified and fixed by
`--seed`.

```bash
python -u precompute_codec.py \
  --data-root /path/to/kinetics/train \
  --codec h264 \
  --qps 30 35 40 45 \
  --codec-io pipe \
  --codec-workers 2 \
  --output-dir precomputed_codec/h264
```

Then distill a codec-specific proxy without invoking FFmpeg in every epoch:

```bash
python -u train_proxy.py \
  --precomputed-root precomputed_codec/h264 \
  --codec h264 \
  --qps 30 35 40 45 \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 20 \
  --batch-size 8 \
  --hidden-channels 48 \
  --latent-channels 64 \
  --bottleneck-channels 96 \
  --blocks-per-stage 2 \
  --film-channels 64 \
  --qp-step-divisor 12 \
  --clip-grad 1.0 \
  --scheduler-factor 0.5 \
  --scheduler-patience 3 \
  --output-dir checkpoints/h264_proxy
```

Every cached training batch is balanced across the four QPs. Batch sizes 8 or
16 are recommended. Validation always evaluates the fixed cached split. The
legacy online path remains available by replacing `--precomputed-root` with
`--data-root`; it now uses raw pipes and two codec workers by default.

The proxy architecture changed, so a shallow-proxy `last.pt` cannot be resumed.
Start FiLM deeper-3D training at epoch 1 with a new output directory. Existing
precomputed codec caches remain fully reusable because their real reconstruction
and BPP targets are architecture-independent.

Then train Video Swin Lite through the real codec and frozen proxy:

```bash
python -u train.py \
  --data-root /path/to/kinetics/train \
  --proxy-checkpoint checkpoints/h264_proxy/best.pt \
  --preprocessor swin \
  --swin-patch-size 4 \
  --swin-embed-dim 48 \
  --swin-depth 4 \
  --swin-heads 4 \
  --swin-window-temporal 4 \
  --swin-window-spatial 8 \
  --codec h264 \
  --codec-qps 30 35 40 45 \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 30 \
  --batch-size 1 \
  --accumulation-steps 4 \
  --output-dir checkpoints/preprocessor
```

If `val/` is absent, training creates a deterministic stratified validation subset
using indices in memory. Classes containing only one video remain in training.

## Real-codec evaluation

Final evaluation never uses the proxy. It compares the anchor and preprocessed
clips through the real FFmpeg codec and frozen analyzer. If `val/` is absent,
`evaluate_real_codec.py` automatically recreates the checkpoint's stratified
validation split in memory from its saved `val_ratio` and `seed`; no validation
folder, symlinks or precomputed codec cache are required.

```bash
python -u evaluate_real_codec.py \
  --checkpoint checkpoints/preprocessor/best.pt \
  --data-root /path/to/kinetics/train \
  --codecs h264 \
  --qps 30 35 40 45 \
  --device cuda \
  --output-dir outputs/real_codec
```

The output includes `metrics.csv`, `metrics.json`, `bd_rate.json`, and one
`<codec>_top1_bpp_bd_rate.png` plot per codec. Task BD-rate uses Top-1 as the
quality axis; PSNR BD-rate is also reported. Negative BD-rate means bitrate
saving at equal quality. Task BD-rate is reported as undefined when discrete
Top-1 curves have too few distinct points or no overlapping accuracy range.

Omit `--limit` for the final result. `--limit 200` is useful for a faster pilot,
but produces noisier Top-1 and task BD-rate estimates. The limit applies only to
evaluation videos after the deterministic split and is independent of the proxy
precompute limits.

## Kaggle and model summary

Ready-to-run Kaggle cells are in [KAGGLE_GUIDE_VI.md](KAGGLE_GUIDE_VI.md).

```bash
python model_summary.py \
  --model all \
  --preprocessor swin \
  --frames 16 \
  --frame-size 128 \
  --device auto
```

## Main files

- `preprocessing/swin.py`: Video Swin Lite and 3-D shifted-window attention.
- `preprocessing/model.py`: preprocessor factory plus factorized ViT/CNN ablations.
- `preprocessing/standard_codec.py`: FFmpeg codecs, proxy and gradient bridge.
- `precompute_codec.py`: deterministic train/val uint8 codec cache and pipe verification.
- `train_proxy.py`: distill the proxy from real codec outputs and measured BPP.
- `train.py`: train only the preprocessor with rate-distortion-task loss.
- `preprocessing/evaluation.py`: reproducible held-out split and BD-rate helpers.
- `model_summary.py`: torchinfo summaries for preprocessor and proxy.
- `evaluate_real_codec.py`: real-codec metrics, Top-1/BPP plots and BD-rate.
- `visualize_pipeline.py`: qualitative output from the same held-out split.
