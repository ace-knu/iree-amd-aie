"""corr vs fp32 BERT on the 64 held-out sentences, real-token positions only.
Usage: eval.py <model.onnx>...   (fp32 reference: ../bert_base.onnx or $BERT_FP32)"""
import sys, numpy as np, onnxruntime as ort
from common import fp32_reference, corr_per_sentence
ref = fp32_reference()
for p in sys.argv[1:]:
    c = corr_per_sentence(ort.InferenceSession(p, providers=["CPUExecutionProvider"]), ref)
    print(f"{p.split('/')[-1]:32s} corr mean {c.mean():.4f} (min {c.min():.4f})")
