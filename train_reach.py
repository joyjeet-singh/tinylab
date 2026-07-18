"""
train_reach.py -- train one reachability arm per its recipe card.

Usage:  python train_reach.py --config configs/reach_flat_structured.yaml --seed 0

Mirrors train.py's house rules: seed everything, write a manifest before any
work (conditions only, never results), log one measurement per line as it
happens, and fingerprint the data so the scorekeeper can prove every arm saw
exactly the same questions.
"""
import argparse
import hashlib

import torch
import yaml

import lablog
from seed import set_seed

try:
    from reachability import graphs, model as M
except ImportError:
    import graphs
    import model as M


def data_fingerprint(instances):
    """Content hash of a dataset: same questions <-> same hash, always."""
    h = hashlib.sha256()
    for inst in instances:
        edges = sorted(tuple(sorted(e)) for e in inst.edges)
        h.update(repr((inst.n_nodes, edges, inst.query,
                       inst.reachable, inst.distance)).encode())
    return h.hexdigest()


def evaluate(net, data, n_nodes, eval_loops, batch_size=256):
    """Exam on held-out questions, broken down by the things that hide in an average."""
    net.eval()
    by_d_correct, by_d_total = {}, {}
    ur_correct = ur_total = 0
    with torch.no_grad():
        for s in range(0, len(data), batch_size):
            chunk = data[s:s + batch_size]
            batch = M.collate(chunk, n_nodes)
            pred = (torch.sigmoid(net(batch, n_loops=eval_loops)["reach_logit"]) > 0.5).long()
            for j, inst in enumerate(chunk):
                right = int(pred[j].item() == inst.reachable)
                if inst.reachable:
                    by_d_correct[inst.distance] = by_d_correct.get(inst.distance, 0) + right
                    by_d_total[inst.distance] = by_d_total.get(inst.distance, 0) + 1
                else:
                    ur_correct += right; ur_total += 1
    net.train()
    acc_by_distance = {d: by_d_correct[d] / by_d_total[d] for d in sorted(by_d_total)}
    reach_correct = sum(by_d_correct.values()); reach_total = sum(by_d_total.values())
    overall = (reach_correct + ur_correct) / (reach_total + ur_total)
    return {
        "test_accuracy": round(overall, 6),
        "acc_reachable": round(reach_correct / reach_total, 6) if reach_total else None,
        "acc_unreachable": round(ur_correct / ur_total, 6) if ur_total else None,
        "acc_by_distance": {int(d): round(a, 6) for d, a in acc_by_distance.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = args.seed
    set_seed(cfg["seed"])

    d, t, m = cfg["data"], cfg["training"], cfg["model"]
    N = d["n_nodes"]

    train = graphs.generate_dataset(d["n_train"], d["train_distances"], N,
                                    data_seed=d["data_seed"])
    test = graphs.generate_dataset(d["n_test"], d["test_distances"], N,
                                   data_seed=d["data_seed"] + 1)

    run_dir, log, close = lablog.start_run(
        cfg, tag=m["name"],
        extra={"train_data_sha256": data_fingerprint(train),
               "test_data_sha256": data_fingerprint(test),
               "n_nodes": N})

    net = M.ReachabilityModel(encoder=m["encoder"], reasoning=m["reasoning"],
                              width=m["width"], layers=m["layers"], n_nodes=N)
    opt = torch.optim.Adam(net.parameters(), lr=t["learning_rate"])
    g = torch.Generator().manual_seed(cfg["seed"])   # run seed shuffles batches + loop counts

    step = 0
    for epoch in range(t["epochs"]):
        order = torch.randperm(len(train), generator=g).tolist()
        for s in range(0, len(order), t["batch_size"]):
            chunk = [train[i] for i in order[s:s + t["batch_size"]]]
            batch = M.collate(chunk, N)
            T = int(torch.randint(1, t["max_train_loops"] + 1, (1,), generator=g).item())
            out = net(batch, n_loops=T)
            loss, r, fr = M.loss_fn(out, batch, frontier_weight=t["frontier_weight"])
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 20 == 0:
                log({"kind": "train", "epoch": epoch, "step": step,
                     "loss": round(loss.item(), 6), "train_loops": T})

        ev = evaluate(net, test, N, eval_loops=t["eval_loops"])
        ev.update({"kind": "eval", "epoch": epoch, "step": step})
        log(ev)
        print(f"epoch {epoch}: overall {ev['test_accuracy']:.4f}  "
              f"reach {ev['acc_reachable']:.4f}  unreach {ev['acc_unreachable']:.4f}")

    log({"kind": "done", "total_steps": step})
    close()
    print("wrote", run_dir)


if __name__ == "__main__":
    main()
