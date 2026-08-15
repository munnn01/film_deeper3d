# Video Swin Lite preprocessing through standard video codecs

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

First distill a codec-specific proxy:

```bash
python -u train_proxy.py \
  --data-root /path/to/kinetics/train \
  --codec h264 \
  --qps 30 35 40 45 50 \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 20 \
  --output-dir checkpoints/h264_proxy
```

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
  --codec-qps 30 35 40 45 50 \
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
- `train_proxy.py`: distill the proxy from real codec outputs and measured BPP.
- `train.py`: train only the preprocessor with rate-distortion-task loss.
- `model_summary.py`: torchinfo summaries for preprocessor and proxy.
- `evaluate_real_codec.py`: anchor/preprocessed rate-accuracy evaluation.
- `visualize_pipeline.py`: qualitative pipeline output.
