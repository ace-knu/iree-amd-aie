"""Shared by eval.py / loo.py: held-out sentences and the fp32 reference."""
import os, numpy as np, onnxruntime as ort
HERE = os.path.dirname(os.path.abspath(__file__))
FP32 = os.environ.get("BERT_FP32", os.path.join(HERE, "..", "bert_base.onnx"))
EVAL_IDS = np.load(os.path.join(HERE, "eval_ids.npy"))

def fp32_reference():
    s = ort.InferenceSession(FP32, providers=["CPUExecutionProvider"])
    return [s.run(None, {s.get_inputs()[0].name: r[None]})[0][0] for r in EVAL_IDS]

def corr_per_sentence(session, ref):
    """corr vs fp32 over the real-token positions ([PAD] = 0 excluded) of each sentence."""
    out = []
    for r, t in zip(EVAL_IDS, ref):
        n = int((r != 0).sum()); y = session.run(None, {"input_ids": r[None]})[0][0]
        out.append(np.corrcoef(y[:n].ravel().astype(np.float64), t[:n].ravel().astype(np.float64))[0, 1])
    return np.array(out)
