"""Step 2. The obvious rule: each rider gets the nearest driver still free.

Riders are served in the order they asked: rider A first, then B, and so on.
A driver that is taken is gone, so a later rider chooses from what is left.
"""
import numpy as np

from city import BLOCK_SECONDS, cost_table, total
from step1_city import RIDERS, snapshot


def nearest_driver(cost):
    """cost[rider][driver] = blocks. Returns the pairs [(rider, driver)]."""
    pairs, taken = [], set()
    for rider in range(len(cost)):
        free = [d for d in range(len(cost[0])) if d not in taken]
        if not free:
            break                         # no drivers left: the rest wait
        driver = min(free, key=lambda d: cost[rider][d])
        taken.add(driver)
        pairs.append((rider, driver))
    return pairs


def many_snapshots(n=6, count=1000, seed=0):
    """count random moments with n riders and n drivers: a table each."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10 ** 9, count)
    return [cost_table(*snapshot(n, n, int(s))) for s in seeds]


def run():
    riders, drivers = snapshot()
    cost = cost_table(riders, drivers)
    pairs = nearest_driver(cost)
    for rider, driver in pairs:
        wanted, away = int(np.argmin(cost[rider])), cost[rider][driver]
        note = f'   (driver {wanted + 1} was nearer, but taken)'
        print(f'rider {RIDERS[rider]} gets driver {driver + 1}: '
              f'{away:>2} blocks = {away * BLOCK_SECONDS:>3} s'
              + note * (wanted != driver))
    blocks = total(cost, pairs)
    print(f'total wait: {blocks} blocks = {blocks * BLOCK_SECONDS} s '
          f'({blocks * BLOCK_SECONDS / len(pairs):.1f} s per rider)')

    # the same rule on 1,000 random moments: who pays for being served late?
    tables = many_snapshots()
    by_turn = np.zeros(6)
    for c in tables:
        for rider, driver in nearest_driver(c):
            by_turn[rider] += c[rider][driver] * BLOCK_SECONDS / len(tables)
    print('1,000 random moments, average wait by turn: ' + '  '.join(
        f'{RIDERS[r]} {by_turn[r]:.0f} s' for r in range(6)))
    print(f'the last rider waits {by_turn[5] / by_turn[0]:.1f} times as long '
          f'as the first; average {by_turn.mean():.1f} s per rider')
    return {'pairs': pairs, 'blocks': blocks, 'by_turn': by_turn}


if __name__ == '__main__':
    run()
