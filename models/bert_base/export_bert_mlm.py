"""Export bert-base-uncased WITH its masked-language-model head to ONNX, so
the compiled model produces a human-checkable answer instead of just a raw
hidden-state tensor. This is what models/bert_base/demo_mlm.py runs against.

BertForMaskedLM = BertModel (see export_bert_base.py) + BertOnlyMLMHead
(Linear(768,768) -> GELU -> LayerNorm -> Linear(768,vocab_size), the last
Linear tied to the input word embeddings). The extra Linear(768,vocab_size)
is just another matmul+bias, same as every other Linear in the encoder --
nothing new needed in the NPU/CPU split for it.

Unlike export_bert_base.py, this one takes attention_mask as a real input
(not omitted): the demo pads variable-length real sentences to SEQ_LEN, and
padding must not be attended to or it corrupts the real tokens' output
through self-attention. The extra masking-prep ops this adds (Equal, Where,
etc. -- see docs/2026-08-19_bert_tiny_e2e.md) are all non-contractions, so
they land on the host either way; the only cost is a slightly larger graph.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM

MODEL_ID = "bert-base-uncased"
SEQ_LEN = 32  # keep in the seq_len>=32 safe zone (see docs/2026-08-19_bert_tiny_e2e.md)
OUT = "bert_mlm.onnx"

model = BertForMaskedLM.from_pretrained(MODEL_ID, attn_implementation="eager")
model.eval()

# The vocab-projection matmul is [*, 768] x [768, vocab_size=30522], and
# 30522 is not a multiple of 8 -- the AIE instruction size requires M/N/K to
# all divide the (8, 8, 8) tile shape, so amd-aie refuses to compile it
# ("has element types which must target an AIE instruction size that does
# not divide ... N (30522) ..."), same class of check as the seq_len=17/20
# repros in docs/2026-08-19_bert_tiny_e2e.md. Pad the decoder weight/bias up
# to the next multiple of 8 (30528) with zeros purely for tracing -- this
# doesn't touch the real model parameters (a fresh padded tensor is built
# here, not assigned back), so nothing else changes. demo_mlm.py slices the
# output back to [:VOCAB_SIZE] before reading it, so the padded columns
# (whose logits are meaningless zeros) are never considered.
VOCAB_SIZE = model.config.vocab_size
PAD_VOCAB = -VOCAB_SIZE % 8
PADDED_VOCAB_SIZE = VOCAB_SIZE + PAD_VOCAB
with torch.no_grad():
    decoder_weight = F.pad(model.cls.predictions.decoder.weight, (0, 0, 0, PAD_VOCAB)).detach()
    decoder_bias = F.pad(model.cls.predictions.decoder.bias, (0, PAD_VOCAB)).detach()

# Placeholder input purely for tracing shapes -- the compiled model doesn't
# depend on which sentence this is; demo_mlm.py substitutes real sentences
# against the already-compiled .vmfb at run time.
input_ids = torch.zeros((1, SEQ_LEN), dtype=torch.int64)
attention_mask = torch.ones((1, SEQ_LEN), dtype=torch.int64)


class MLMOnly(torch.nn.Module):
    def __init__(self, m, decoder_weight, decoder_bias):
        super().__init__()
        self.m = m
        self.decoder_weight = decoder_weight
        self.decoder_bias = decoder_bias

    def forward(self, input_ids, attention_mask):
        sequence_output = self.m.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        transformed = self.m.cls.predictions.transform(sequence_output)
        return F.linear(transformed, self.decoder_weight, self.decoder_bias)


wrapped = MLMOnly(model, decoder_weight, decoder_bias)
with torch.no_grad():
    ref = wrapped(input_ids, attention_mask)

torch.onnx.export(
    wrapped, (input_ids, attention_mask), OUT,
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    opset_version=17, dynamic_axes=None, dynamo=False,
)
print(f"saved {OUT}, logits shape {tuple(ref.shape)}")
