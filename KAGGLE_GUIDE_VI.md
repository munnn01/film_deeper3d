# Các cell Kaggle chạy trực tiếp

Bật GPU và Internet cho Kaggle Notebook, sửa `DATA` nếu dataset được gắn ở đường
dẫn khác, sau đó chọn **Run All**.

## Cell 1 — Clone và cài đặt

```python
%cd /kaggle/working
!git clone -q https://github.com/munnn01/film_deeper3d.git
%cd /kaggle/working/film_deeper3d
%pip install -q --no-cache-dir -r requirements.txt
```

## Cell 2 — Đường dẫn

```python
DATA = "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train"
PROJECT = "/kaggle/working/film_deeper3d"
PROXY_DIR = "/kaggle/working/checkpoints/h264_film_deeper3d"
CACHE_DIR = "/kaggle/working/precomputed_codec/h264"
MODEL_DIR = "/kaggle/working/checkpoints/video_swin_lite"
EVAL_DIR = "/kaggle/working/real_codec_eval"
VIS_DIR = "/kaggle/working/visualization"
```

`DATA` ở đây trỏ trực tiếp tới thư mục chứa các thư mục lớp. Không cần có `val/`;
hai script train tự tạo validation split phân tầng trong bộ nhớ.

## Cell 3 — Model summary

```python
!python "$PROJECT/model_summary.py" \
  --model all \
  --preprocessor swin \
  --swin-patch-size 4 \
  --swin-embed-dim 48 \
  --swin-depth 4 \
  --swin-heads 4 \
  --swin-window-temporal 4 \
  --swin-window-spatial 8 \
  --qp 35 \
  --frames 16 \
  --frame-size 128 \
  --device auto
```

## Cell 4 — Pre-compute codec xác định, tách train/val

Cell này chỉ chạy FFmpeg một lần cho mỗi clip/QP. Clip gốc được lưu đúng một bản
`uint8`; reconstruction của bốn QP nằm riêng trong `train/` và `val/`. Raw pipe
được so pixel/BPP với đường PNG cũ trước khi cache và dùng đúng 2 FFmpeg workers.

```python
!python -u "$PROJECT/precompute_codec.py" \
  --data-root "$DATA" \
  --codec h264 \
  --qps 30 35 40 45 \
  --fps 30 \
  --preset medium \
  --codec-io pipe \
  --codec-workers 2 \
  --ffmpeg-threads 1 \
  --verify-pipe \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --val-ratio 0.1 \
  --seed 42 \
  --output-dir "$CACHE_DIR"
```

Cache `uint8` gồm một clip gốc và bốn reconstruction nên có thể lớn hơn dữ liệu
video nén. Khi quota `/kaggle/working` không đủ, thêm `--limit-train N` và
`--limit-val M`, hoặc dùng một Kinetics subset nhỏ hơn.

## Cell 5 — Distill H.264 proxy từ cache

```python
!python -u "$PROJECT/train_proxy.py" \
  --precomputed-root "$CACHE_DIR" \
  --codec h264 \
  --qps 30 35 40 45 \
  --fps 30 \
  --preset medium \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 20 \
  --batch-size 8 \
  --workers 4 \
  --hidden-channels 48 \
  --latent-channels 64 \
  --bottleneck-channels 96 \
  --blocks-per-stage 2 \
  --film-channels 64 \
  --qp-step-divisor 12 \
  --clip-grad 1.0 \
  --scheduler-factor 0.5 \
  --scheduler-patience 3 \
  --amp \
  --output-dir "$PROXY_DIR"
```

`--batch-size 8` tạo mỗi batch gồm cân bằng cả bốn QP; có thể tăng lên 16 nếu GPU
đủ VRAM. Checkpoint lưu cả optimizer, AMP scaler và scheduler để resume đúng.
Proxy shallow cũ không tương thích với kiến trúc này, vì vậy phải train từ epoch 1
với `PROXY_DIR` mới. Cache codec đã tạo trước đây vẫn dùng lại được.

## Cell 6 — Train Video Swin Lite preprocessor

```python
!python -u "$PROJECT/train.py" \
  --data-root "$DATA" \
  --proxy-checkpoint "$PROXY_DIR/best.pt" \
  --preprocessor swin \
  --swin-patch-size 4 \
  --swin-embed-dim 48 \
  --swin-depth 4 \
  --swin-heads 4 \
  --swin-window-temporal 4 \
  --swin-window-spatial 8 \
  --max-residual 0.25 \
  --codec h264 \
  --codec-qps 30 35 40 45 \
  --codec-fps 30 \
  --codec-preset medium \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 30 \
  --batch-size 1 \
  --accumulation-steps 4 \
  --workers 4 \
  --amp \
  --output-dir "$MODEL_DIR"
```

## Cell 7 — Đánh giá codec thật

```python
!python -u "$PROJECT/evaluate_real_codec.py" \
  --checkpoint "$MODEL_DIR/best.pt" \
  --test-dir "$DATA" \
  --codecs h264 \
  --qps 30 35 40 45 \
  --device cuda \
  --output-dir "$EVAL_DIR"
```

Do `DATA` trỏ trực tiếp tới `train/`, cell này đánh giá toàn bộ thư mục đó. Validation
metrics của held-out split tự động vẫn được lưu trong checkpoint khi train.

## Cell 8 — Trực quan hóa Top-1

```python
!python -u "$PROJECT/visualize_pipeline.py" \
  --checkpoint "$MODEL_DIR/best.pt" \
  --data-root "$DATA" \
  --sample-index 0 \
  --codec h264 \
  --codec-qp 35 \
  --device cuda \
  --output-dir "$VIS_DIR"
```

File JSON và tiêu đề ảnh/video chỉ hiển thị dự đoán và accuracy Top-1, không tạo
danh sách Top-5.

## Cell 9 — Nén kết quả

```python
%cd /kaggle/working
!zip -qr film_deeper3d_results.zip checkpoints real_codec_eval visualization
```

## Cell 10 — Link tải xuống

```python
from IPython.display import FileLink

FileLink("/kaggle/working/film_deeper3d_results.zip")
```
