"""Local duplicate-match harness: python sim.py HERO OPP... [--deals N].
Names: 'model' (model.Model), dummy names, sparring bot names."""
import random
import sys
import time
from multiprocessing import Pool

from arena.dummies import DUMMIES
from card_utils import full_deck
from engine import play_hand


def make(name, seed):
    if name == "model":
        from model import Model
        return Model()
    if name.startswith("mod:"):
        mod, cls = name[4:].split(".")
        import old_v1
        import variants
        m = {"old_v1": old_v1, "variants": variants}[mod]
        return m.__dict__[cls]()
    from sparring_bots import SPARRING_BOTS
    table = dict(DUMMIES)
    table.update(SPARRING_BOTS)
    return table[name](seed=seed)


def match(args):
    a, b, deals, seed = args
    net_a = 0
    faults = 0
    slow = 0.0
    for rot in range(2):
        pa, pb = make(a, seed), make(b, seed + 1)
        for d in range(deals):
            deck = full_deck()
            random.Random(f"{seed}:{d}").shuffle(deck)
            seats = [pa, pb] if rot == 0 else [pb, pa]
            names = [a, b] if rot == 0 else [b, a]
            t = time.perf_counter()
            r = play_hand(seats, deck, [500, 500], 0, 10, 20, names)
            slow = max(slow, time.perf_counter() - t)
            ia = 0 if rot == 0 else 1
            net_a += r.net[ia]
            if ia in r.faults:
                faults += 1
                if faults <= 3:
                    print("FAULT", a, r.faults[ia], flush=True)
        for m in (pa, pb):
            if m.__dict__.get("errors"):
                print("MODEL ERRORS", a, b, m.errors, flush=True)
    hands = 2 * deals
    return a, b, net_a / 20 / hands * 100, faults, slow


if __name__ == "__main__":
    argv = sys.argv[1:]
    deals = 300
    seeds = 1
    if "--deals" in argv:
        i = argv.index("--deals")
        deals = int(argv[i + 1])
        del argv[i:i + 2]
    if "--seeds" in argv:
        i = argv.index("--seeds")
        seeds = int(argv[i + 1])
        del argv[i:i + 2]
    off = 0
    if "--offset" in argv:
        i = argv.index("--offset")
        off = int(argv[i + 1])
        del argv[i:i + 2]
    hero, opps = argv[0], argv[1:]
    jobs = [(hero, o, deals, 1000 * s + 7 + off) for o in opps for s in range(seeds)]
    with Pool(8) as p:
        res = p.map(match, jobs)
    agg = {}
    for a, b, bb100, f, slow in res:
        agg.setdefault(b, []).append((bb100, f, slow))
    tot = 0
    for b, rows in agg.items():
        m = sum(r[0] for r in rows) / len(rows)
        tot += m
        print(f"{hero:>10} vs {b:<12} {m:+8.1f} bb/100  faults={sum(r[1] for r in rows)}  slowest_hand={max(r[2] for r in rows):.2f}s")
    print(f"mean {tot / len(agg):+.1f}")
