"""Export prajjwal1/bert-tiny (2-layer, hidden=128, 2-head BERT) to ONNX for e2e NPU testing.

Fixed shapes (batch=1, seq_len=16) -- the amd-aie pipeline needs static shapes,
same constraint as models/vgg16 (fixed batch=1). Run inside the dev container
venv (torch/onnx are already there); `transformers` is not, and the venv's
site-packages isn't writable by the non-root container user, so install it
into a scratch dir instead of touching the venv:

  source /opt/venv/bin/activate
  pip install --target=/tmp/bert_deps transformers
  PYTHONPATH=/tmp/bert_deps python3 export_bert_tiny.py

No tokenizer: this tests the compute graph end-to-end (matmul on NPU, the rest
on CPU), the same way vgg16's recipe feeds a random image instead of a real
one -- token identity doesn't matter, only that input_ids stay in [0, vocab_size).
This also sidesteps prajjwal1/bert-tiny shipping no fast-tokenizer.json, which
current transformers can no longer auto-convert from its legacy vocab.txt.

Outputs input_ids as the only runtime input (no attention_mask -- there is no
padding to mask with a fixed unpadded sequence, and omitting it avoids extra
mask-prep ops that `attn_implementation="sdpa"`'s masking_utils bakes into the
graph even when the mask is all-ones) and last_hidden_state as the only output
(pooler dropped -- it is just a Linear+Tanh on token 0, adds nothing
architecturally new to test). `attn_implementation="eager"` selects the plain
matmul+softmax attention path instead of SDPA, for a smaller/more predictable
op set.
"""

import numpy as np
import torch
from transformers import BertModel

MODEL_ID = "prajjwal1/bert-tiny"
SEQ_LEN = 32  # 16 hits a amd-aie codegen crash in the Attn@V batched matmul
              # (see docs/2026-08-19 memory / commit notes: M=16,N=64 batched
              # matmuls overflow tile-column allocation); 32 avoids it.
OUT = "bert_tiny.onnx"

# AutoModel/AutoConfig reject this repo's config.json (no `model_type` key --
# it predates that convention); BertModel.from_pretrained parses it directly
# since it already knows the target class.
model = BertModel.from_pretrained(MODEL_ID, attn_implementation="eager")
model.eval()

rng = np.random.default_rng(0)
input_ids = torch.from_numpy(
    rng.integers(0, model.config.vocab_size, size=(1, SEQ_LEN)).astype(np.int64)
)


class EncoderOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return self.m(input_ids=input_ids).last_hidden_state


wrapped = EncoderOnly(model)
with torch.no_grad():
    ref = wrapped(input_ids)

torch.onnx.export(
    wrapped, (input_ids,), OUT,
    input_names=["input_ids"], output_names=["last_hidden_state"],
    opset_version=17, dynamic_axes=None, dynamo=False,
)
np.save("input_ids.npy", input_ids.numpy())
np.save("ref_last_hidden_state.npy", ref.numpy())
print(f"saved {OUT}, shapes: input_ids={tuple(input_ids.shape)} "
      f"last_hidden_state={tuple(ref.shape)}")
