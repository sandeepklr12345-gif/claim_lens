"""Fine-tune a cross-encoder on data/pairs_v2.jsonl (SPEC.md 13.3).

    python -m eval.train_matcher --base <model-id-or-path> --out models\\matcher_v2

Seed 42, fp16, batch 16, max length 256, at most 4 epochs, early stopping on validation F1.
If the GPU runs out of memory: --batch-size 8 --max-length 192 --grad-accum 2.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claimlens.data.phase2 import load_pairs, load_splits
from claimlens.matcher.metrics import best_threshold, f1, roc_auc
from claimlens.matcher.scorer import TRAIN_META, sigmoid
from claimlens.schemas import Pair
from claimlens.utils.config import ROOT

log = logging.getLogger(__name__)
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tokenizer, pairs: list[Pair], max_length: int, device: str):
    return tokenizer([p.reason for p in pairs], [p.clause_text for p in pairs], padding=True,
                     truncation="only_second", max_length=max_length,
                     return_tensors="pt").to(device)


@torch.no_grad()
def predict(model, tokenizer, pairs: list[Pair], max_length: int, device: str) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(pairs), 16):
        enc = encode(tokenizer, pairs[i:i + 16], max_length, device)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            out.append(model(**enc).logits[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def train(args: argparse.Namespace) -> dict:
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = ROOT / "data"
    splits, _ = load_splits(data)
    pairs = load_pairs(data / "pairs_v2.jsonl")
    train_pairs = [p for p in pairs if splits[p.clause_id] == "train"]
    val_pairs = [p for p in pairs if splits[p.clause_id] == "val"]
    y_val = np.array([p.label for p in val_pairs])

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=1, ignore_mismatched_sizes=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    order = list(range(len(train_pairs)))
    rng = random.Random(SEED)
    history: list[dict] = []
    best_key = (-1.0, -1.0)
    best_epoch = 0
    threshold = 0.5
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(order)
        losses = []
        optimizer.zero_grad()
        for step, i in enumerate(range(0, len(order), args.batch_size), start=1):
            batch = [train_pairs[j] for j in order[i:i + args.batch_size]]
            enc = encode(tokenizer, batch, args.max_length, device)
            target = torch.tensor([p.label for p in batch], dtype=torch.float, device=device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(model(**enc).logits[:, 0], target) / args.grad_accum
            scaler.scale(loss).backward()
            losses.append(loss.item() * args.grad_accum)
            if step % args.grad_accum == 0 or i + args.batch_size >= len(order):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        prob = np.asarray(sigmoid(predict(model, tokenizer, val_pairs, args.max_length, device)))
        val_f1 = f1(y_val, (prob >= 0.5).astype(int))
        val_auc = roc_auc(y_val, prob)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_f1_at_0.5": val_f1, "val_roc_auc": val_auc})
        log.info("epoch %d: %s", epoch, history[-1])
        if (val_f1, val_auc) > best_key:
            best_key, best_epoch = (val_f1, val_auc), epoch
            threshold = best_threshold(y_val, prob)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
        elif epoch - best_epoch >= args.patience:
            log.info("early stop at epoch %d", epoch)
            break

    meta = {"base": args.base, "seed": SEED, "fp16": device == "cuda", "epochs_run": len(history),
            "best_epoch": best_epoch, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "max_length": args.max_length, "lr": args.lr, "n_train": len(train_pairs),
            "n_val": len(val_pairs), "val_threshold": threshold, "history": history,
            "seconds": round(time.time() - start, 1)}
    (out_dir / TRAIN_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="models/matcher_v2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=2)
    meta = train(ap.parse_args())
    log.info("done: best epoch %d of %d, %.0fs", meta["best_epoch"], meta["epochs_run"],
             meta["seconds"])


if __name__ == "__main__":
    main()
