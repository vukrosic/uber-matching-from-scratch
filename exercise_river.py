"""The exercise from the README, solved: a river.

A river runs down the middle of the city, between x = 20 and x = 21. Cars
cross it only on bridges. A driver right across the water looks one block
away, and is really a long detour away.

Nothing in the auction changes. Only the table does: `river_table` gives the
real blocks by road. Two dispatchers run the same 10 second batch on the
same evenings: one fills the table with plain blocks (blind to the river),
one with the real drive. The cars always drive the real roads.
"""
from functools import partial

import numpy as np

from city import BLOCK_SECONDS, cost_table
from step6_batching import average, batch

RIVER_X = 20                # the river runs between x = 20 and x = 21
BRIDGES = (10, 30)          # the streets (y) that cross it
BUSY = 0.7                  # an evening with drivers to spare
EVENINGS = 24
LAYOUTS = {'no river': None, 'three bridges': (5, 20, 35),
           'two bridges': BRIDGES, 'one bridge': (20,)}


def river_table(riders, drivers, bridges=BRIDGES):
    """Blocks by road. Same side: as before. Other side: drive along the
    river to the best bridge, cross it, and drive on."""
    a, b = np.asarray(riders), np.asarray(drivers)
    other_side = (a[:, None, 0] <= RIVER_X) != (b[None, :, 0] <= RIVER_X)
    across = np.abs(a[:, None, 0] - b[None, :, 0])
    ys = np.array(bridges)
    up = (np.abs(a[:, None, None, 1] - ys)
          + np.abs(b[None, :, None, 1] - ys)).min(axis=2)
    return np.where(other_side, across + up, cost_table(a, b))


def run():
    rider, drivers = [(20, 20)], [(21, 20), (14, 20)]
    plain = cost_table(rider, drivers)[0]
    real = river_table(rider, drivers)[0]
    print(f'rider at (20, 20), driver 1 at (21, 20), driver 2 at (14, 20), '
          f'bridges at y = {BRIDGES[0]} and y = {BRIDGES[1]}')
    print(f'  blocks on the map: {plain[0]} and {plain[1]}; '
          f'blocks by road: {real[0]} and {real[1]}')
    print(f'  blind dispatcher sends driver {np.argmin(plain) + 1}: '
          f'{real[np.argmin(plain)] * BLOCK_SECONDS} s; the real table '
          f'sends driver {np.argmin(real) + 1}: {real.min() * BLOCK_SECONDS}'
          f' s')

    print(f'{EVENINGS} evenings at busy {BUSY}, 10 s batch: average wait, '
          f'1 in 20 over, 1 in 100 over')
    rows = {}
    for name, bridges in LAYOUTS.items():
        if bridges is None:
            rows[name] = (average(batch, 10, BUSY, seeds=EVENINGS),) * 2
        else:
            road = partial(river_table, bridges=bridges)
            rows[name] = (
                average(batch, 10, BUSY, road, cost_table, EVENINGS),
                average(batch, 10, BUSY, road, None, EVENINGS))
        blind, sees = rows[name]
        print(f'  {name:<14} blind {blind["wait"]:5.1f} s {blind["p95"]:4.0f}'
              f' s {blind["p99"]:4.0f} s   real table {sees["wait"]:5.1f} s '
              f'{sees["p95"]:4.0f} s {sees["p99"]:4.0f} s   '
              f'{1 - sees["wait"] / blind["wait"]:4.0%} shorter')
    for name in ('two bridges', 'one bridge'):
        blind, sees = rows[name]
        wins = sum(y < x for x, y in zip(blind['each'], sees['each']))
        print(f'{name}: the real table is better on {wins} of {EVENINGS} '
              f'evenings; pickup drive {blind["pickup"]:.0f} -> '
              f'{sees["pickup"]:.0f} s')
    return {'example': (rider, drivers, plain, real), 'rows': rows}


if __name__ == '__main__':
    run()
