"""Tokenize one sentence per line into an `input_ids` npy [N, 32] (bert-base-uncased,
[CLS] ... [SEP] then [PAD]=0). Needs `transformers` (not on the host image):
  pip install --target=/tmp/tokdeps 'transformers<5'
  PYTHONPATH=/tmp/tokdeps python3 tokenize_sentences.py calib_sentences.txt calib_ids.npy"""
import sys, numpy as np
from transformers import AutoTokenizer
src, dst = sys.argv[1:3]; seq = int(sys.argv[3]) if len(sys.argv) > 3 else 32
lines = [l.strip() for l in open(src) if l.strip()]
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
ids = tok(lines, padding="max_length", truncation=True, max_length=seq, return_tensors="np")["input_ids"]
np.save(dst, ids.astype(np.int64)); print(f"{len(lines)} sentences -> {dst} {ids.shape}")
