"""Score NPU outputs from run_npu.sh against fp32 BERT, and optionally against the ORT
reference model of the same stage (same metric as quant/eval.py).

  npu_eval.py <out_dir> [ref.onnx]          # BERT_FP32 = fp32 model (default ../bert_base.onnx)

Prints corr mean (min) over the 64 sentences, real-token positions only, and with a
reference model the NPU - ORT difference in standard errors."""
import os, sys
import numpy as np, onnxruntime as ort
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quant"))
from common import EVAL_IDS, fp32_reference

out = sys.argv[1]
ref = fp32_reference()
def corr(y, t, n): return np.corrcoef(y[:n].ravel().astype(float), t[:n].ravel().astype(float))[0, 1]
npu = []
for i, (r, t) in enumerate(zip(EVAL_IDS, ref)):
    y = np.load(f"{out}/npu_{i:02d}.npy").reshape(t.shape)
    npu.append(corr(y, t, int((r != 0).sum())))
npu = np.array(npu)
print(f"NPU  corr mean {npu.mean():.4f} (min {npu.min():.4f})")
if len(sys.argv) > 2:
    # 2 intra-op threads, as in every ORT number in the docs: the int8 chain is sensitive
    # enough that the thread count (float summation order) moves the min by ~0.002.
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.intra_op_num_threads = 2; so.inter_op_num_threads = 1
    s = ort.InferenceSession(sys.argv[2], so, providers=["CPUExecutionProvider"])
    o = np.array([corr(s.run(None, {"input_ids": r[None]})[0][0], t, int((r != 0).sum()))
                  for r, t in zip(EVAL_IDS, ref)])
    d = npu - o
    print(f"ORT  corr mean {o.mean():.4f} (min {o.min():.4f})")
    print(f"NPU - ORT {d.mean():+.4f} ({d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.1f} SE)")
