"""Export bert-base-uncased (12-layer, hidden=768, 12-head BERT) to ONNX for
e2e NPU testing. Scaled-up sibling of models/bert_tiny -- see
models/bert_tiny/export_bert_tiny.py and docs/2026-08-19_bert_tiny_e2e.md for
why each choice below is made the way it is:

- SEQ_LEN=32, not 16: seq_len=16 crashes iree-compile's tile allocator for
  BERT-tiny's Attn@V matmul (M=N=seq_len=16, N/8=8 N-tiles overflow npu4's
  4-row-per-column limit); 32 was confirmed safe there. bert-base's Attn@V has
  the same M=N=seq_len shape (batch=12 instead of 2, head_dim=64 either way),
  untested at this batch count -- this export is also the first real test of
  whether the row-overflow bug depends on batch size.
- attn_implementation="eager": keeps the exported graph to plain
  MatMul/Softmax/MatMul instead of the SDPA masking-utils op soup.
- No tokenizer, random input_ids in [0, vocab_size): bert-base-uncased does
  ship a real tokenizer (unlike bert-tiny), but this export still only tests
  the compute graph, not language understanding, so there's no need for it
  here either.
"""

import numpy as np
import torch
from transformers import BertModel

MODEL_ID = "bert-base-uncased"
SEQ_LEN = 32
OUT = "bert_base.onnx"

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
