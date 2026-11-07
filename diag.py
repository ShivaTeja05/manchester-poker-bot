"""Where does model win/lose vs an opponent? python diag.py OPP [deals]"""
import random
import sys
from collections import defaultdict
from multiprocessing import Pool

from card_utils import full_deck
from engine import play_hand
from sim import make


def run(args):
    opp, deals, seed = args
    out = defaultdict(lambda: [0.0, 0])
    for rot in range(2):
        m, o = make("model", seed), make(opp, seed + 1)
        for d in range(deals):
            deck = full_deck()
            random.Random(f"{seed}:{d}").shuffle(deck)
            seats = [m, o] if rot == 0 else [o, m]
            r = play_hand(seats, deck, [500, 500], 0, 10, 20, ["model", opp] if rot == 0 else [opp, "model"])
            me = 0 if rot == 0 else 1
            pos = "SB" if me == 0 else "BB"
            board = 0
            street = "pre"
            who_folded = None
            for a in r.actions:
                if a.kind == "reveal_card" and a.player_id == -1:
                    board += 1
                    street = {3: "flop", 4: "turn", 5: "river"}.get(board, street)
                if a.kind == "fold":
                    who_folded = "me" if a.player_id == me else "opp"
            end = f"{street}:{'sd' if who_folded is None else who_folded+'_fold'}"
            for key in (pos, f"{pos}:{end}"):
                out[key][0] += r.net[me] / 20
                out[key][1] += 1
    return dict(out)


if __name__ == "__main__":
    opp = sys.argv[1]
    deals = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    with Pool(8) as p:
        parts = p.map(run, [(opp, deals, 1000 * s + 7) for s in range(8)])
    tot = defaultdict(lambda: [0.0, 0])
    for part in parts:
        for k, (v, n) in part.items():
            tot[k][0] += v
            tot[k][1] += n
    hands = tot["SB"][1] + tot["BB"][1]
    for k in sorted(tot):
        v, n = tot[k]
        print(f"{k:<22} n={n:5d} ({n / hands * 100:5.1f}%)  total={v / hands * 100:+7.1f} bb/100  avg={v / n:+6.2f} bb")
