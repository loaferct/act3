"""Search over maintenance manuals, SOPs and work-order history (BM25, no external services)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_STOP = set("the a an and or of to in on at for is are be by with from as it this that if not no into per when then".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Chunk:
    id: str
    doc: str
    section: str
    text: str


class KnowledgeBase:
    def __init__(self, folder: str | Path):
        self.chunks: list[Chunk] = []
        for path in sorted(Path(folder).glob("*.md")):
            title, section, buf = path.stem, None, []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                elif line.startswith("## "):
                    self._add(title, section, buf)
                    section, buf = line[3:].strip(), []
                else:
                    buf.append(line)
            self._add(title, section, buf)
        self._docs = [tokenize(f"{c.doc} {c.section} {c.section} {c.text}") for c in self.chunks]
        self._tf = [Counter(d) for d in self._docs]
        n = len(self._docs)
        df = Counter(t for d in self._docs for t in set(d))
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._avg = sum(map(len, self._docs)) / max(n, 1)

    def _add(self, title, section, buf):
        text = "\n".join(buf).strip()
        if section and text:
            self.chunks.append(Chunk(f"KB{len(self.chunks) + 1}", title, section, text))

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        q = tokenize(query)
        scores = []
        for i, (tf, doc) in enumerate(zip(self._tf, self._docs)):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self._idf[t] * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(doc) / self._avg))
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.chunks[i] for i in order[:k] if scores[i] > 0]

    def get(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.id == chunk_id), None)
