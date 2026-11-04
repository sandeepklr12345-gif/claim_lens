"""CLI: python -m claimlens.rag.search "waiting period" --doc care_supreme"""

from __future__ import annotations

import argparse
import logging

from claimlens.rag.parse import list_policies
from claimlens.rag.retriever import build_retriever


def main() -> None:
    ap = argparse.ArgumentParser(description="Search a policy for clauses")
    ap.add_argument("query")
    ap.add_argument("--doc", required=True, help=f"policy id, one of {list_policies()}")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    retriever = build_retriever(args.doc, dense=args.mode != "bm25")
    for r in retriever.retrieve(args.query, args.k, mode=args.mode):
        c = r.chunk
        print(f"[{c.chunk_id}] p.{c.page}  {c.section_path}")
        print(f"    {c.text[:220]}...")
        print(f"    rrf={r.score:.4f} bm25_rank={r.bm25_rank} dense_rank={r.dense_rank}")


if __name__ == "__main__":
    main()
