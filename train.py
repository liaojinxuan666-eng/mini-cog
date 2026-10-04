import time
import torch
import torch.nn.functional as F

from config import CFG
from model import HybridLM
from data import make_copy_batch


def build_model(cfg, kind="hybrid", device="cuda"):
    if kind == "hybrid":
        attn_layer = cfg.model.attn_layer
    elif kind == "ssm":
        attn_layer = -1          # 永不命中，全部走 SSM
    elif kind == "attn":
        attn_layer = None        # 见下方说明
    else:
        raise ValueError(f"unknown kind: {kind}")

    return HybridLM(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_head=cfg.model.n_head,
        window=cfg.model.window,
        n_layer=cfg.model.n_layer,
        attn_layer=attn_layer,
        max_len=cfg.model.max_len,
        dropout=cfg.model.dropout,
    ).to(device)


def train_one(kind="hybrid", steps=None, cfg=CFG):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    steps = steps or cfg.train.steps

    torch.manual_seed(cfg.train.seed)
    model = build_model(cfg, kind=kind, device=device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    half_len = cfg.train.half_len
    t0 = time.time()
    losses = []

    for step in range(1, steps + 1):
        model.train()
        x, y = make_copy_batch(
            cfg.train.batch_size, half_len,
            cfg.model.vocab_size, device,
        )
        logits = model(x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.model.vocab_size),
            y[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()

        losses.append(loss.item())

        if step % cfg.train.log_every == 0:
            avg = sum(losses[-cfg.train.log_every:]) / cfg.train.log_every
            print(f"[{kind}] step {step}/{steps}  loss {avg:.4f}  "
                  f"elapsed {time.time()-t0:.0f}s")

    # 最终评估
    model.eval()
    with torch.no_grad():
        x, y = make_copy_batch(
            cfg.train.batch_size, half_len,
            cfg.model.vocab_size, device,
        )
        logits = model(x)
        pred = logits[:, half_len:-1].argmax(-1)
        target = y[:, half_len + 1:]
        acc = (pred == target).float().mean().item()

    print(f"[{kind}] params={n_params:,}  final_acc={acc:.4f}  "
          f"total_time={time.time()-t0:.0f}s")
    return {"kind": kind, "params": n_params, "acc": acc,
            "losses": losses, "time": time.time() - t0}


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else None
    train_one(kind=kind, steps=steps)