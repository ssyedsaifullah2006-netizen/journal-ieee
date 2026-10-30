"""
One-time privacy migration for the released YouTube corpus.

Replaces every commenter display name in the released data files with an irreversible
pseudonym ("user_" + 8 hex chars). A random salt is generated for the run and discarded,
so pseudonyms cannot be reversed by hashing guessed handles, but the mapping is consistent
across all files (the same commenter has the same pseudonym everywhere).

Comment IDs are deliberately left unchanged so that records can still be re-verified
against the platform. Display names are not used by any estimator in this repository.

Usage (from the repository root):  python experiments/pseudonymize_authors.py
"""
import hashlib
import json
import secrets
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS = DATA / "youtube_corpus_100_comments.csv"
LABELED = DATA / "youtube_corpus_100_comments_human_labeled.csv"
LABELING_JSON = DATA / "comments_for_labeling.json"
MANIFEST = CORPUS.with_suffix(".provenance.json")


def main():
    salt = secrets.token_bytes(32)
    cache = {}

    def pseudo(handle):
        handle = str(handle)
        if handle.startswith("user_") and len(handle) == 13:
            return handle  # already pseudonymized
        if handle not in cache:
            cache[handle] = "user_" + hashlib.sha256(salt + handle.encode("utf-8")).hexdigest()[:8]
        return cache[handle]

    original_sha = hashlib.sha256(CORPUS.read_bytes()).hexdigest()

    df = pd.read_csv(CORPUS)
    df["author"] = df["author"].map(pseudo)
    df.to_csv(CORPUS, index=False)

    lab = pd.read_csv(LABELED)
    lab["author"] = lab["author"].map(pseudo)
    lab.to_csv(LABELED, index=False)

    items = json.loads(LABELING_JSON.read_text(encoding="utf-8"))
    for it in items:
        it["author"] = pseudo(it["author"])
    LABELING_JSON.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.setdefault("original_csv_sha256", manifest.get("csv_sha256", original_sha))
    manifest["csv_sha256"] = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    note = ("commenter display names replaced by irreversible pseudonyms (user_XXXXXXXX) "
            "after retrieval; comment text, IDs, likes, ranks and timestamps unchanged")
    manifest["post_processing"] = (manifest.get("post_processing", "") + "; " + note).lstrip("; ")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Pseudonymized {len(cache)} distinct display names across 3 files; manifest updated.")


if __name__ == "__main__":
    main()
