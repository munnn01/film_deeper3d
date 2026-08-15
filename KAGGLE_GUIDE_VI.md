# Các cell Kaggle chạy trực tiếp

Bật GPU và Internet cho Kaggle Notebook, sửa `DATA` nếu dataset được gắn ở đường
dẫn khác, sau đó chọn **Run All**.

## Cell 1 — Clone và cài đặt

```python
%cd /kaggle/working
!git clone -q https://github.com/munnn01/preprocessor_Video_Swin_Lite_proxy.git
%cd /kaggle/working/preprocessor_Video_Swin_Lite_proxy
%pip install -q --no-cache-dir -r requirements.txt
```

## Cell 2 — Đường dẫn

```python
DATA = "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train"
PROJECT = "/kaggle/working/preprocessor_Video_Swin_Lite_proxy"
PROXY_DIR = "/kaggle/working/checkpoints/h264_proxy"
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

## Cell 4 — Distill H.264 proxy

```python
!python -u "$PROJECT/train_proxy.py" \
  --data-root "$DATA" \
  --codec h264 \
  --qps 30 35 40 45 50 \
  --fps 30 \
  --preset medium \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 20 \
  --batch-size 2 \
  --workers 4 \
  --amp \
  --output-dir "$PROXY_DIR"
```

## Cell 5 — Train Video Swin Lite preprocessor

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
  --codec-qps 30 35 40 45 50 \
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

## Cell 6 — Đánh giá codec thật

```python
!python -u "$PROJECT/evaluate_real_codec.py" \
  --checkpoint "$MODEL_DIR/best.pt" \
  --test-dir "$DATA" \
  --codecs h264 \
  --qps 30 35 40 45 50 \
  --device cuda \
  --output-dir "$EVAL_DIR"
```

Do `DATA` trỏ trực tiếp tới `train/`, cell này đánh giá toàn bộ thư mục đó. Validation
metrics của held-out split tự động vẫn được lưu trong checkpoint khi train.

## Cell 7 — Trực quan hóa

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

## Cell 8 — Nén kết quả

```python
%cd /kaggle/working
!zip -qr video_swin_lite_proxy_results.zip checkpoints real_codec_eval visualization
```

## Cell 9 — Link tải xuống

```python
from IPython.display import FileLink

FileLink("/kaggle/working/video_swin_lite_proxy_results.zip")
```
