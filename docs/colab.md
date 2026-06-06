# Training on Google Colab (GPU)

This project's tiny model does **not** require a powerful GPU — a free Colab **T4**
is more than enough, and even CPU works (just slower). An H100 would be idle at
this scale. Use Colab mainly for convenience/speed.

> Reminder: the model is trained **from scratch**. Do not load any pretrained
> weights. Only use the project's own `src/train.py`.

## 0. Set the runtime to GPU
`Runtime -> Change runtime type -> Hardware accelerator: GPU`.

## 1. Get the code (zip + Google Drive)
We upload a zip of the project to Google Drive, then unzip it inside Colab.

**On your machine** — a clean `text-project.zip` is produced by:

```bash
# from the project root; excludes venv/git/outputs/large books
rm -f text-project.zip && zip -r -q text-project.zip . \
  -x '.git/*' '.venv/*' 'outputs/*' 'data/processed/*' '.cursor/*' '.cache/*' \
     '.pytest_cache/*' '*/__pycache__/*' '__pycache__/*' '*.pyc' '.DS_Store' '*/.DS_Store' \
     'data/raw/alice.txt' 'data/raw/holmes.txt' 'data/raw/grimm.txt' 'data/raw/poe.txt' \
     'text-project.zip'
```

Then upload `text-project.zip` to your Google Drive (e.g. to `MyDrive/`).

**In Colab** — mount Drive and unzip:

```python
from google.colab import drive
drive.mount('/content/drive')

# Adjust the path if you put the zip somewhere other than the Drive root.
!cp "/content/drive/MyDrive/text-project.zip" /content/
!cd /content && unzip -o -q text-project.zip -d text-project
%cd /content/text-project
!ls
```

> The zip excludes the large book files and outputs to stay small; you re-download
> the books in step 3 below. (Alternatively, you can skip the downloader and upload
> your own `data/processed/corpus.txt` to Drive the same way.)

## 2. Install dependencies
Colab already ships PyTorch with CUDA. Install the few extras this project uses:

```python
!pip install -q pyyaml matplotlib
import torch
print("CUDA available:", torch.cuda.is_available(), "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
```

## 3. Get the data and prepare the corpus
```python
!python scripts/download_gutenberg.py --books alice holmes grimm poe --output_dir data/raw
!python scripts/prepare_corpus.py --input_dir data/raw --output_path data/processed/corpus.txt
```

## 4. Train
The config's `device: auto` will automatically select CUDA on Colab.

```python
# Full run as configured (3000 iters):
!python -m src.train --config configs/tiny_transformer.yaml --corpus_path data/processed/corpus.txt
```

Because a GPU is fast, you may optionally train longer for better text (this only
changes iteration count, not the architecture):

```python
!python -m src.train --config configs/tiny_transformer.yaml --corpus_path data/processed/corpus.txt --max_iters_override 8000
```

## 5. Plot the loss curve
```python
!python -m src.plot_metrics --metrics_path outputs/metrics/losses.csv --output_path outputs/metrics/loss_curve.png
from IPython.display import Image
Image("outputs/metrics/loss_curve.png")
```

## 6. Try generation
```python
!python -m src.generate --checkpoint outputs/checkpoints/ckpt.pt --prompt "The " --max_new_tokens 500 --temperature 0.8 --top_k 40
```

## 7. Download the artifacts back to your machine
```python
from google.colab import files
files.download("outputs/metrics/losses.csv")
files.download("outputs/metrics/loss_curve.png")
files.download("outputs/checkpoints/ckpt.pt")
files.download("outputs/checkpoints/tokenizer.json")
```

After downloading, place them back under `outputs/` locally and paste the real
numbers (final train/val loss, a sample) into `docs/experiment_log.md`.
