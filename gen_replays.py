"""Plays model vs sparring bots and writes replay JSON for the viewer.
python gen_replays.py OUT.json [deals_per_opponent]"""
import json
import random
import sys

from card_utils import full_deck
from engine import play_hand
from sim import make
import variants


class Rec(variants.LowR if "--lowr" in sys.argv else __import__("model").Model):
    def seed(self, hand, player_id):
        super().seed(hand, player_id)
        self.trace = []

    def run(self, actions, game_state):
        self.debug = None
        a = super().run(actions, game_state)
        self.trace.append(self.debug)
        return a


def frames_for(result, names, bot_seat, trace):
    stacks = [500, 500]
    bets = [10, 20]
    stacks[0] -= 10
    stacks[1] -= 20
    pot = 30
    board = []
    shown = {0: [], 1: []}
    out = [{"t": f"Blinds: {names[0]} posts 10 (SB, button), {names[1]} posts 20 (BB)",
            "actor": None, "pot": pot, "stacks": list(stacks), "bets": list(bets),
            "board": [], "think": None}]
    ti = 0
    street = "preflop"
    for a in result.actions:
        p = a.player_id
        think = None
        if a.kind == "reveal_card":
            if p == -1:
                board.append(str(a.card))
                ns = {3: "flop", 4: "turn", 5: "river"}.get(len(board))
                if ns and ns != street:
                    street = ns
                    bets = [0, 0]
                if len(board) in (1, 2):
                    continue  # show the flop as one step
                text = f"Dealer: {street} " + " ".join(board[-3:] if len(board) == 3 else board[-1:])
            else:
                shown[p].append(str(a.card))
                if len(shown[p]) < 2:
                    continue
                text = f"{names[p]} shows {' '.join(shown[p])}"
        elif a.kind == "win":
            text = f"{names[p]} wins {a.amount}"
        else:
            if p == bot_seat and ti < len(trace):
                think = trace[ti]
                ti += 1
            owe = max(bets) - bets[p]
            if a.kind == "fold":
                text = f"{names[p]} folds"
            elif a.kind == "call":
                add = min(owe, stacks[p])
                stacks[p] -= add
                bets[p] += add
                pot += add
                text = f"{names[p]} " + ("checks" if add == 0 else f"calls {add}" + (" (all-in)" if stacks[p] == 0 else ""))
            else:
                add = a.amount - bets[p]
                stacks[p] -= add
                bets[p] = a.amount
                pot += add
                verb = "bets" if owe == 0 and street != "preflop" else "raises to"
                text = f"{names[p]} {verb} {a.amount}" + (" (all-in)" if stacks[p] == 0 else "")
        out.append({"t": text, "actor": p, "pot": pot, "stacks": list(stacks),
                    "bets": list(bets), "board": list(board), "think": think,
                    "street": street})
    return out


def main():
    out_path = sys.argv[1]
    deals = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 25
    opps = ["tag", "lag", "potodds", "nash", "shover50", "station", "minraiser", "equity"]
    hands = []
    for oi, opp in enumerate(opps):
        for rot in range(2):
            bot = Rec()
            other = make(opp, 77 + oi)
            for d in range(deals):
                deck = full_deck()
                random.Random(f"replay:{opp}:{d}").shuffle(deck)
                if rot == 0:
                    seats, names, bot_seat = [bot, other], ["OurBot", opp], 0
                else:
                    seats, names, bot_seat = [other, bot], [opp, "OurBot"], 1
                r = play_hand(seats, deck, [500, 500], 0, 10, 20, names)
                hands.append({
                    "opp": opp, "rot": rot, "deal": d, "names": names, "bot": bot_seat,
                    "hole": [[str(c) for c in h] for h in r.hands],
                    "board": [str(c) for c in r.board],
                    "net": r.net, "frames": frames_for(r, names, bot_seat, bot.trace),
                })
    with open(out_path, "w") as f:
        json.dump(hands, f, separators=(",", ":"))
    print(len(hands), "hands")


if __name__ == "__main__":
    main()
