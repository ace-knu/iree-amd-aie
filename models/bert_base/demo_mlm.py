"""demo_mlm.py: sentence-in, word-out demo for the compiled bert_mlm_trunk.vmfb.

This is the ONLY verified-correct MLM demo path in this repo. The tempting
alternative -- compiling the full model (encoder + MLM head, including the
[768x30528] vocab-projection matmul) and running it entirely on npu4 -- was
tried on 2026-08-19 (see export_bert_mlm.py) and found unusable: it compiles
fine, and once the NPU dispatch timeout is raised past its 2s default (that
matmul overruns it -- `ert state 8` / TIMEOUT) it also *runs* without error,
but produces a DIFFERENT wrong answer on every run of the identical compiled
model against the identical input (verified: two consecutive runs of the
same .vmfb/input differed by up to 21.5 in raw logit values, with completely
different top-5 words each time). That's not a precision issue -- simulating
bf16 rounding for that same matmul on the host reproduces the correct answer
fine -- it's a genuine non-determinism/correctness bug specific to this
matmul's scale (`N=30528` is far larger than anything else compiled in this
repo; the next-largest is the FFN's `N=3072`). Root cause not investigated
further; see models/bert_base/README.md's "MLM demo" section for the full
writeup. Don't try to "fix" this by re-running bert_mlm.vmfb hoping for a
good result -- it's not a timing fluke to retry past, it's wrong by design
until someone root-causes it.

This script instead stops the compiled graph at `transformed`, the
`[1,32,768]` output of `BertPredictionHeadTransform` (nothing bigger than the
already-reliable FFN `768x3072` matmuls), and does the final
`transformed @ decoder_weight.T + decoder_bias` projection in plain numpy on
the host CPU -- trivial cost for a `[32,768]x[768,30522]` matmul, and it's
the version that actually gives correct answers.

Usage (inside the dev container, with bert_mlm_trunk.vmfb already built --
see export_bert_mlm_trunk.py):

    PYTHONPATH=/tmp/bert_deps python3 models/bert_base/demo_mlm.py \\
        "The capital of France is [MASK]."

Prints the top-5 predicted words for the [MASK] position. Requires exactly
one [MASK] token.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

MODEL_ID = "bert-base-uncased"
SEQ_LEN = 32
HERE = Path(__file__).parent
VMFB = HERE / "bert_mlm_trunk.vmfb"
DECODER_WEIGHT = HERE / "decoder_weight.npy"  # [vocab_size, 768], from export_bert_mlm_trunk.py
DECODER_BIAS = HERE / "decoder_bias.npy"      # [vocab_size]
IREE_RUN_MODULE = Path(__file__).resolve().parents[2] / "build/tools/iree-run-module"


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} '<sentence with exactly one [MASK]>'")
    sentence = sys.argv[1]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    enc = tokenizer(
        sentence, return_tensors="np", padding="max_length", truncation=True,
        max_length=SEQ_LEN,
    )
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)

    mask_positions = np.where(input_ids[0] == tokenizer.mask_token_id)[0]
    if len(mask_positions) != 1:
        sys.exit(f"expected exactly one {tokenizer.mask_token!r}, found {len(mask_positions)}")
    mask_pos = mask_positions[0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.save(tmp / "input_ids.npy", input_ids)
        np.save(tmp / "attention_mask.npy", attention_mask)
        out_path = tmp / "transformed.npy"
        subprocess.run(
            [
                str(IREE_RUN_MODULE), "--device=amdxdna", "--device=local-task",
                f"--module={VMFB}", "--function=main_graph",
                f"--input=@{tmp / 'input_ids.npy'}",
                f"--input=@{tmp / 'attention_mask.npy'}",
                f"--output=@{out_path}",
            ],
            check=True,
        )
        transformed = np.load(out_path)  # [1, SEQ_LEN, 768], from npu4

    # Final vocab projection on the host -- see module docstring.
    decoder_weight = np.load(DECODER_WEIGHT)  # [vocab_size, 768]
    decoder_bias = np.load(DECODER_BIAS)      # [vocab_size]
    logits = transformed[0, mask_pos] @ decoder_weight.T + decoder_bias

    top5 = np.argsort(-logits)[:5]
    print(f"Input:  {sentence}")
    print("Top-5 predictions for [MASK]:")
    for rank, token_id in enumerate(top5, 1):
        word = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
        print(f"  {rank}. {word}")
    filled = sentence.replace(tokenizer.mask_token, tokenizer.convert_ids_to_tokens([int(top5[0])])[0], 1)
    print(f"Output: {filled}")


if __name__ == "__main__":
    main()
