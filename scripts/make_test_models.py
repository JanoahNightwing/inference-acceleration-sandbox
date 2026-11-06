"""Write two tiny random-weight checkpoints in llama2.c format for smoke-testing the engine.

The draft model is the base model plus noise, so speculative decoding sees partial acceptance.
Output is gibberish text -- use the real tinyllamas checkpoints for meaningful generations.
"""
import struct, sys
from pathlib import Path
import numpy as np

def write(path, w, cfg):
    dim, hidden, layers, heads, kv_heads, vocab, seq_len = cfg
    with open(path, "wb") as f:
        f.write(struct.pack("7i", dim, hidden, layers, heads, kv_heads, vocab, seq_len))  # vocab>0 => shared classifier
        for arr in w:
            f.write(arr.astype(np.float32).tobytes())

def main(out_dir="models", noise=0.35, seed=0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dim, hidden, layers, heads, vocab, seq_len = 64, 172, 2, 4, 32000, 256
    cfg = (dim, hidden, layers, heads, heads, vocab, seq_len)
    hs = dim // heads
    shapes = [(vocab, dim), (layers, dim), (layers, dim, dim), (layers, dim, dim), (layers, dim, dim),
              (layers, dim, dim), (layers, dim), (layers, hidden, dim), (layers, dim, hidden),
              (layers, hidden, dim), (dim,), (seq_len, hs // 2), (seq_len, hs // 2)]
    base = []
    for i, s in enumerate(shapes):
        if len(s) == 1 or (len(s) == 2 and i in (1, 6)):
            base.append(np.ones(s))                     # rmsnorm weights
        elif i == 0:
            base.append(rng.normal(0, 0.6, s))          # embeddings / shared classifier
        else:
            base.append(rng.normal(0, 1 / np.sqrt(s[-1]), s))
    draft = [a + rng.normal(0, noise * a.std() if a.std() > 0 else 0, a.shape) for a in base]
    write(out / "test-base.bin", base, cfg)
    write(out / "test-draft.bin", draft, cfg)
    print(f"wrote {out/'test-base.bin'} and {out/'test-draft.bin'}")

if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["models"]))
