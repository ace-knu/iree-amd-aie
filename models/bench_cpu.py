"""Plain-PyTorch CPU inference benchmark, for comparing against the npu4
numbers in models/bert_tiny/README.md and models/bert_base/README.md.

Self-contained on purpose -- no dependency on this repo's build, IREE, or
the amd-aie plugin. Only needs `pip install torch transformers`. Runs the
same shape (batch=1, seq_len=32) as the NPU recipes, with the same
attn_implementation="eager" and random input_ids (no tokenizer needed --
this measures the compute graph, not language understanding, matching how
export_bert_tiny.py/export_bert_base.py build their inputs).

Paste this whole file into a Colab cell (or `!python bench_cpu.py` after
uploading it) to get a CPU number from a different machine than the npu4
host this repo's numbers were measured on.

Usage:
    python3 bench_cpu.py                          # benchmarks both models
    python3 bench_cpu.py prajjwal1/bert-tiny       # just one
    python3 bench_cpu.py bert-base-uncased 64      # override seq_len
"""

import sys
import time

import numpy as np
import torch
from transformers import BertModel

SEQ_LEN_DEFAULT = 32
WARMUP_ITERS = 3
TIMED_ITERS = 20


def bench(model_id, seq_len):
    print(f"\n=== {model_id} (seq_len={seq_len}) ===")
    model = BertModel.from_pretrained(model_id, attn_implementation="eager")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M")

    rng = np.random.default_rng(0)
    input_ids = torch.from_numpy(
        rng.integers(0, model.config.vocab_size, size=(1, seq_len)).astype(np.int64)
    )

    torch.set_num_threads(torch.get_num_threads())  # uses all available cores by default
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            model(input_ids=input_ids)

        latencies = []
        for _ in range(TIMED_ITERS):
            start = time.perf_counter()
            model(input_ids=input_ids)
            latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies)
    print(f"latency over {TIMED_ITERS} runs: "
          f"mean={latencies.mean()*1000:.1f}ms  "
          f"median={np.median(latencies)*1000:.1f}ms  "
          f"min={latencies.min()*1000:.1f}ms  max={latencies.max()*1000:.1f}ms")
    return latencies


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
        seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else SEQ_LEN_DEFAULT
        bench(model_id, seq_len)
    else:
        bench("prajjwal1/bert-tiny", SEQ_LEN_DEFAULT)
        bench("bert-base-uncased", SEQ_LEN_DEFAULT)
