"""How do our all-ins do? python allin_diag.py OPP... (in-process, duplicate)."""
import random
import sys
from collections import defaultdict
from multiprocessing import Pool

from card_utils import full_deck
from engine import play_hand
from sim import make
from model import Model


class Rec(Model):
    def seed(self, hand, player_id):
        super().seed(hand, player_id)
        self.shoves = []

    def run(self, actions, game_state):
        self.debug = None
        a = super().run(actions, game_state)
        if a.kind == "raise" and a.amount == game_state.max_raise_to and self.debug:
            self.shoves.append((self.street, self.debug[0]))
        return a


def job(args):
    opp, seed, deals = args
    out = defaultdict(lambda: [0, 0, 0.0])  # count, opp_folded, net_bb
    for rot in range(2):
        m, o = Rec(), make(opp, seed + 1)
        for d in range(deals):
            deck = full_deck()
            random.Random(f"{seed}:{d}").shuffle(deck)
            seats = [m, o] if rot == 0 else [o, m]
            r = play_hand(seats, deck, [500, 500], 0, 10, 20, ["m", opp] if rot == 0 else [opp, "m"])
            me = 0 if rot == 0 else 1
            if not m.shoves:
                continue
            street, eq = m.shoves[0]
            kind = ("pre" if street == 0 else "post") + ("_bluff" if eq < 0.45 else "_value")
            folded = any(a.kind == "fold" and a.player_id != me for a in r.actions)
            out[kind][0] += 1
            out[kind][1] += folded
            out[kind][2] += r.net[me] / 20
    return opp, dict(out)


if __name__ == "__main__":
    opps = sys.argv[1:]
    with Pool(10) as p:
        res = p.map(job, [(o, 1000 * s + 4242, 150) for o in opps for s in range(3)])
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))
    for opp, d in res:
        for k, (n, f, net) in d.items():
            a = agg[opp][k]
            a[0] += n; a[1] += f; a[2] += net
    for opp in opps:
        for k in ("pre_value", "pre_bluff", "post_value", "post_bluff"):
            n, f, net = agg[opp][k]
            if n:
                print(f"{opp:<10} {k:<11} shoves={n:4d} opp_folds={f / n:5.0%} avg={net / n:+6.2f} bb total={net:+8.1f} bb")
