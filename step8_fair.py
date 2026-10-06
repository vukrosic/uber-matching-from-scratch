"""Step 8. The shortest total is not fair to everyone.

When riders outnumber free drivers, the auction hands each driver to a rider
close by. A rider far from where trips end is never the cheap choice, so
that rider can be passed over batch after batch.

The fix changes the table, not the auction: every 60 seconds a rider has
already waited takes one block off that rider's row. A long wait makes every
driver look nearer to that rider, until one of them is sent.

Measured on evenings 10% busier than the rush evening, where drivers run
out for longer.
"""
import numpy as np

from city import clock
from step4_auction import best_match
from step5_evening import make_requests, simulate, waits
from step6_batching import average, batch

WINDOW = 10
BUSIER = 1.1
WORTH = 60      # seconds already waited that count as one block of driving
SWEEP = (None, 240, 120, 60, 30, 15, 5)
EVENINGS = 24


def fair(cost, waited, worth=WORTH):
    """The same auction on a changed table: waiting buys blocks."""
    return best_match(cost - (waited // worth)[:, None])


def fair_rule(worth):
    if worth in (None, WORTH):
        return batch if worth is None else fair
    return lambda cost, waited: fair(cost, waited, worth)


def run():
    requests = make_requests(0, BUSIER)
    plain, _ = simulate(requests, batch, WINDOW)
    better, _ = simulate(requests, fair, WINDOW)
    w, w2 = waits(plain), waits(better)
    who = int(np.argmax(w))
    t, x, y = requests[who, :3]
    passed = int(((requests[:, 0] > t) & (plain[:, 1] < plain[who, 1])).sum())
    print(f'one busier evening, 10 s batch: average wait {w.mean():.0f} s, '
          f'but the rider at ({x}, {y}) who asked at {clock(t)} waited '
          f'{w[who]} s ({w[who] / 60:.0f} min)')
    print(f'  that rider was passed over in {(plain[who, 1] - t) // WINDOW} '
          f'batches; {passed} riders asked later and were matched earlier')
    print(f'  with waiting counted: that rider waits {w2[who]} s, the worst '
          f'wait of the evening is {w2.max()} s, the average '
          f'{w2.mean():.0f} s')

    print(f'{EVENINGS} busier evenings: average wait, worst wait, riders '
          f'over 10 minutes per evening')
    rows = {}
    for worth in SWEEP:
        r = rows[worth] = average(fair_rule(worth), WINDOW, BUSIER,
                                  seeds=EVENINGS)
        name = f'{worth:>3} s waited = 1 block' if worth else 'not counted'
        print(f'  {name:<22} {r["wait"]:4.0f} s   {r["worst"]:5.0f} s   '
              f'{r["over10"]:5.1f}')
    a, b = rows[None], rows[WORTH]
    wins = sum(y < x for x, y in zip(a['worst_each'], b['worst_each']))
    print(f'at {WORTH} s: worst wait {1 - b["worst"] / a["worst"]:.0%} '
          f'shorter (shorter on {wins} of {EVENINGS} evenings), average wait '
          f'{b["wait"] / a["wait"] - 1:.1%} longer')
    a = average(batch, WINDOW, seeds=EVENINGS)
    b = average(fair, WINDOW, seeds=EVENINGS)
    print(f'the rush evenings of step 7: average {a["wait"]:.0f} -> '
          f'{b["wait"]:.0f} s, worst {a["worst"]:.0f} -> {b["worst"]:.0f} s,'
          f' over 10 minutes {a["over10"]:.1f} -> {b["over10"]:.1f} riders')
    return {'requests': requests, 'plain': plain, 'fair': better,
            'who': who, 'passed': passed, 'rows': rows, 'rush': (a, b)}


if __name__ == '__main__':
    run()
