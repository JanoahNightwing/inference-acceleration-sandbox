"""Simulated speculative decoding that emits the same events as the C engine.

Used when no checkpoints are present (or when the client asks for it), so the UI can be
demoed anywhere. Every run is labelled `source: "simulated"`. Timings follow a
memory-bound accelerator profile: drafting a token is ~8x cheaper than a target forward,
and verifying k tokens in one batch costs about one target forward.
"""
from __future__ import annotations

import asyncio
import random
import re
from typing import AsyncIterator

from .engine import RunParams

STORIES = [
    "Once upon a time, there was a little girl named Lily. She loved to play in the park with her red ball. "
    "One sunny day, Lily saw a small dog sitting under a big tree. The dog looked sad and lonely. "
    "Lily walked over and said, \"Hello, little dog! Do you want to play with me?\" The dog wagged its tail "
    "and barked happily. They played catch all afternoon until the sun went down. When it was time to go home, "
    "the dog followed Lily to her house. Her mom smiled and said the dog could stay. From that day on, "
    "Lily and the dog were best friends, and they played in the park every day.",
    "Once upon a time, in a quiet village, there lived a boy named Tom. Tom had a shiny blue boat that he "
    "loved very much. Every morning he took the boat to the pond near his house. One day, the wind was very "
    "strong and the boat floated far away. Tom was sad and started to cry. A friendly duck swam over and "
    "pushed the boat back to the shore with its beak. Tom was so happy that he gave the duck some bread. "
    "He said, \"Thank you, duck! You are a good friend.\" The duck quacked and swam in a circle. After that, "
    "Tom and the duck played together at the pond every morning.",
    "Once upon a time, there was a big brown bear who lived in the forest. The bear liked to eat honey more "
    "than anything else. One day, he found a tall tree with a beehive at the top. He tried to climb the tree, "
    "but the branches were too thin. The bear sat down and thought very hard. Then he saw a little bird "
    "flying nearby. The bird said, \"I can help you, but you must share.\" The bear agreed, and the bird "
    "asked the bees to give them some honey. The bear and the bird ate the sweet honey together. They learned "
    "that sharing makes everything better.",
]

TOKEN_RE = re.compile(r"\s?[A-Za-z]+|\s?\d+|\s?[^\sA-Za-z\d]")

EASY = {" the", " a", " to", " and", ".", ",", " was", " of", " in", " with", " her", " his", " he", " she",
        " day", " upon", " time", " there", " that", " they", "\"", " said", " very", " named"}
CONFUSIONS = {
    " little": [" small", " big", " young"], " big": [" large", " tall", " small"], " happy": [" glad", " sad", " excited"],
    " sad": [" upset", " scared", " happy"], " park": [" garden", " yard", " forest"], " dog": [" cat", " puppy", " bird"],
    " played": [" ran", " jumped", " walked"], " saw": [" found", " heard", " met"], " loved": [" liked", " wanted", " had"],
    " friend": [" buddy", " pal", " helper"], " boat": [" toy", " ship", " ball"], " pond": [" lake", " river", " park"],
    " bear": [" lion", " fox", " bunny"], " honey": [" food", " apples", " berries"], " tree": [" bush", " house", " hill"],
    " smiled": [" laughed", " nodded", " said"], " friends": [" buddies", " happy", " together"],
}

# memory-bound accelerator profile (microseconds)
DRAFT_TOKEN_US = 120.0
TARGET_TOKEN_US = 1000.0
VERIFY_BASE_US = 1050.0
VERIFY_PER_TOKEN_US = 30.0


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _story_for(prompt: str, rng: random.Random) -> list[str]:
    p = prompt.strip()
    for s in STORIES:
        if p and s.startswith(p):
            return tokenize(s[len(p):])
    return tokenize(" " + rng.choice(STORIES).split(", ", 1)[-1])


def _wrong_guess(true_tok: str, vocab: list[str], rng: random.Random) -> str:
    if true_tok in CONFUSIONS:
        return rng.choice(CONFUSIONS[true_tok])
    for _ in range(10):
        cand = rng.choice(vocab)
        if cand != true_tok:
            return cand
    return " the"


def _jitter(rng: random.Random, v: float, frac: float = 0.08) -> float:
    return round(v * rng.uniform(1 - frac, 1 + frac), 1)


async def simulate(params: RunParams, pace_s: float = 0.002) -> AsyncIterator[dict]:
    rng = random.Random(params.seed or None)
    story = _story_for(params.prompt, rng)
    vocab = sorted(set(story))
    k = params.n_draft
    # higher temperature -> the draft and target disagree more often
    temp_penalty = max(0.0, params.temperature - 0.7) * 0.25
    cfg = {"dim": 288, "hidden_dim": 768, "n_layers": 6, "n_heads": 6, "n_kv_heads": 6, "vocab_size": 32000, "seq_len": 256}
    big = {**cfg, "dim": 768, "hidden_dim": 2048, "n_layers": 12, "n_heads": 12, "n_kv_heads": 12, "seq_len": 1024}
    yield {"type": "start", "engine": "simulator", "source": "simulated", "base": big, "draft": cfg,
           "base_name": "simulated-110M", "draft_name": "simulated-15M",
           "n_draft": k, "steps": params.steps, "temperature": params.temperature, "topp": params.topp,
           "seed": params.seed}
    yield {"type": "prompt", "n_tokens": len(tokenize(params.prompt)) + 1, "pieces": tokenize(params.prompt),
           "prefill_us": _jitter(rng, 4200.0)}

    pos, step, produced, total_drafted, total_accepted, gen_us = 0, 0, 0, 0, 0, 0.0
    limit = min(params.steps, len(story))
    while produced < limit:
        n_draft = min(k, limit - produced - 1)
        drafts, results = [], []
        diverged = False
        t = 0.0
        for i in range(n_draft):
            true_tok = story[pos + i]
            p_ok = (0.97 if true_tok in EASY else 0.80) - temp_penalty
            correct = not diverged and rng.random() < p_ok
            if diverged:   # the draft keeps writing plausibly from its own (wrong) context
                text = story[pos + i] if rng.random() < 0.6 else _wrong_guess(true_tok, vocab, rng)
            else:
                text = true_tok if correct else _wrong_guess(true_tok, vocab, rng)
            diverged = diverged or not correct
            q = round(rng.uniform(0.35, 0.97), 6)
            t += _jitter(rng, DRAFT_TOKEN_US)
            drafts.append((text, correct, q))
            yield {"type": "draft", "step": step, "i": i, "id": -1, "text": text, "q": q, "t_us": round(t, 1)}
            await asyncio.sleep(pace_s)

        accepted = 0
        for i, (text, correct, q) in enumerate(drafts):
            if correct and accepted == i:
                p = round(min(0.999, q * rng.uniform(0.9, 1.35)), 6)
                results.append({"i": i, "id": -1, "status": "accepted", "p": p, "q": q})
                accepted += 1
            elif accepted == i:
                results.append({"i": i, "id": -1, "status": "rejected", "p": round(q * rng.uniform(0.0, 0.3), 6), "q": q})
            else:
                results.append({"i": i, "id": -1, "status": "discarded"})

        nxt = story[pos + accepted]
        draft_us = round(t, 1)
        verify_us = _jitter(rng, VERIFY_BASE_US + VERIFY_PER_TOKEN_US * n_draft) if n_draft else 0.0
        base_token_us = _jitter(rng, TARGET_TOKEN_US)
        step_us = round(draft_us + (verify_us if n_draft else base_token_us), 1)
        yield {"type": "verify", "step": step, "pos": pos, "drafted": n_draft, "accepted": accepted,
               "results": results,
               "next": {"id": -1, "text": nxt, "source": "base" if n_draft == 0 else ("bonus" if accepted == n_draft else "resample")},
               "draft_us": draft_us, "verify_us": verify_us, "base_token_us": base_token_us, "step_us": step_us}
        await asyncio.sleep(pace_s)

        pos += accepted + 1
        produced += accepted + 1
        total_drafted += n_draft
        total_accepted += accepted
        gen_us += step_us
        step += 1

    yield {"type": "done", "steps": step, "tokens": produced, "drafted": total_drafted, "accepted": total_accepted,
           "gen_us": round(gen_us, 1), "total_us": round(gen_us + 4200.0, 1)}
