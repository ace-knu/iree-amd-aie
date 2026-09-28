"""Export bert-base-uncased's MLM trunk (encoder + prediction-head transform,
everything up to but NOT including the final vocab-projection matmul) to
ONNX. Sibling of export_bert_mlm.py -- see that file for why the MLM head
exists and why vocab_size needed padding.

Why this file exists: the full model (encoder + transform + the
[768 x 30528] decoder matmul) compiles fine, but *running* it on real npu4
hardware hits the NPU dispatch watchdog (`ert state 8` = TIMEOUT) on the
decoder matmul -- by far the largest single matmul tried in this repo, and
raising the timeout requires a kernel module reload
(`options amdxdna timeout_in_sec=60` in /etc/modprobe.d/, then
`modprobe -r/modprobe`) which needs interactive sudo not available in this
environment. This is a real hardware/timeout limit, not a correctness bug --
see docs/2026-08-19_bert_tiny_e2e.md and models/vgg16/README.md's own
"ert state 6" note for the same class of issue.

Workaround: stop the compiled graph at `transformed` (the [1,32,768] output
of BertPredictionHeadTransform: Linear+GELU+LayerNorm), which only contains
matmuls already proven to run fine (nothing bigger than the FFN's 768x3072).
demo_mlm.py does the final `transformed @ decoder_weight.T + decoder_bias`
projection in plain numpy on the host CPU after running this on the NPU --
a [32,768]x[768,30522] matmul is trivial for a CPU BLAS call (well under a
second), so nothing is lost in practice; only the very last matmul moves off
the NPU, and only because of the timeout, not because it can't run there.
"""

import numpy as np
import torch
from transformers import BertForMaskedLM

MODEL_ID = "bert-base-uncased"
SEQ_LEN = 32
OUT = "bert_mlm_trunk.onnx"

model = BertForMaskedLM.from_pretrained(MODEL_ID, attn_implementation="eager")
model.eval()

# Saved once for demo_mlm.py to do the final projection on the host --
# tiny compared to the .onnx/.vmfb (a [30522,768] + [30522] float32 array).
with torch.no_grad():
    np.save("decoder_weight.npy", model.cls.predictions.decoder.weight.numpy())
    np.save("decoder_bias.npy", model.cls.predictions.decoder.bias.numpy())

input_ids = torch.zeros((1, SEQ_LEN), dtype=torch.int64)
attention_mask = torch.ones((1, SEQ_LEN), dtype=torch.int64)


class MLMTrunk(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        sequence_output = self.m.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return self.m.cls.predictions.transform(sequence_output)


wrapped = MLMTrunk(model)
with torch.no_grad():
    ref = wrapped(input_ids, attention_mask)

torch.onnx.export(
    wrapped, (input_ids, attention_mask), OUT,
    input_names=["input_ids", "attention_mask"], output_names=["transformed"],
    opset_version=17, dynamic_axes=None, dynamo=False,
)
print(f"saved {OUT}, transformed shape {tuple(ref.shape)}")
