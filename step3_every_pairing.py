"""Step 3. Is the obvious rule the best? Try every pairing and see.

A tiny trap first: two riders, two drivers, and the nearest-driver rule
doubles the wait. Then brute force: write down every way to hand out the
drivers, add up each one, keep the smallest. It always finds the best. It
also stops being possible very quickly: the number of pairings explodes.
"""
import time
from itertools import permutations
from math import factorial

from city import BLOCK_SECONDS, cost_table, how_long, total
from step1_city import snapshot
from step2_nearest_driver import many_snapshots, nearest_driver

TRAP_RIDERS = [(4, 4), (7, 4)]     # A asks first, B a moment later
TRAP_DRIVERS = [(6, 4), (1, 4)]    # driver 1 between them, driver 2 far left


def every_pairing(cost):
    """The best pairs, and how many pairings were tried.
    Needs riders <= drivers."""
    n, m = len(cost), len(cost[0])
    cost = [[int(v) for v in row] for row in cost]   # plain lists: faster
    best, best_total, tried = None, float('inf'), 0
    for drivers in permutations(range(m), n):   # drivers[r]: for rider r
        tried += 1
        blocks = sum(cost[r][drivers[r]] for r in range(n))
        if blocks < best_total:
            best, best_total = drivers, blocks
    return list(enumerate(best)), tried


def run():
    trap = cost_table(TRAP_RIDERS, TRAP_DRIVERS)
    obvious, (best, tried) = nearest_driver(trap), every_pairing(trap)
    print(f'the trap: nearest driver {total(trap, obvious)} blocks = '
          f'{total(trap, obvious) * BLOCK_SECONDS} s; best of {tried} '
          f'pairings {total(trap, best)} blocks = '
          f'{total(trap, best) * BLOCK_SECONDS} s')

    cost = cost_table(*snapshot())
    obvious, (best, tried) = nearest_driver(cost), every_pairing(cost)
    saved = (total(cost, obvious) - total(cost, best)) * BLOCK_SECONDS
    print(f'the six riders of step 2: nearest driver {total(cost, obvious)} '
          f'blocks, best of {tried} pairings {total(cost, best)} blocks '
          f'({saved} s saved)')

    tables = many_snapshots()
    obvious = [total(c, nearest_driver(c)) for c in tables]
    best = [total(c, every_pairing(c)[0]) for c in tables]
    worse = sum(o > b for o, b in zip(obvious, best))
    better = sum(o < b for o, b in zip(obvious, best))
    print(f'1,000 random moments with 6 riders: the obvious rule misses the '
          f'best pairing in {worse / 10:.1f}% of them and beats it in '
          f'{better}')
    print(f'average total: obvious {sum(obvious) / 1000 * BLOCK_SECONDS:.0f}'
          f' s, best {sum(best) / 1000 * BLOCK_SECONDS:.0f} s '
          f'({sum(obvious) / sum(best) - 1:.1%} longer)')

    rows = []
    for n in range(2, 11):
        c = cost_table(*snapshot(n, n, seed=n))
        t0 = time.process_time()
        _, tried = every_pairing(c)
        rows.append((n, tried, time.process_time() - t0))
        print(f'{n:>2} riders: {tried:>9,} pairings   '
              f'{rows[-1][2] * 1000:>8.1f} ms')
    speed = rows[-1][1] / rows[-1][2]     # pairings per second, measured
    estimates = [(n, factorial(n) / speed) for n in (12, 15, 20, 30)]
    for n, seconds in estimates:
        print(f'{n} riders: {factorial(n):.1e} pairings   '
              f'estimated {how_long(seconds)}')
    return {'trap': trap, 'rows': rows, 'speed': speed, 'worse': worse,
            'better': better, 'estimates': estimates,
            'obvious': sum(obvious) / 1000, 'best': sum(best) / 1000}


if __name__ == '__main__':
    run()
