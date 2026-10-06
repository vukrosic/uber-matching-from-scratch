"""Step 1. A city on a grid, riders, drivers, and how long a pickup takes.

Streets run left-right and up-down. Everyone stands on a street corner,
written as (x, y). A car cannot cut through buildings, so the distance
between two corners is the blocks across plus the blocks up. One block takes
15 seconds.

The whole problem lives in one table: a row for every rider, a column for
every driver, and in each cell the blocks that driver would have to drive.
"""
import numpy as np

from city import BLOCK_SECONDS, SIZE, blocks, cost_table, random_points

NEIGHBOURHOOD = 10      # the pictured examples use a 10 x 10 corner
RIDERS = 'ABCDEFGHIJ'   # riders get letters, drivers get numbers


def snapshot(n_riders=6, n_drivers=6, seed=6, size=NEIGHBOURHOOD):
    """One frozen moment: where the riders and the free drivers are."""
    rng = np.random.default_rng(seed)
    riders = random_points(n_riders, rng, size)
    drivers = random_points(n_drivers, rng, size)
    return riders, drivers


def print_table(cost):
    heads = ''.join(f'driver {d + 1}  ' for d in range(len(cost[0])))
    print('         ' + heads)
    for r, row in enumerate(cost):
        print(f'rider {RIDERS[r]}  ' + ''.join(f'{v:>6}    ' for v in row))


def run():
    print(f'the city: {SIZE} x {SIZE} blocks, one block takes '
          f'{BLOCK_SECONDS} s; corner to corner: '
          f'{2 * SIZE * BLOCK_SECONDS // 60} min')
    riders, drivers = snapshot()
    a, d = riders[0], drivers[0]
    across, up = abs(a[0] - d[0]), abs(a[1] - d[1])
    print(f'rider A at ({a[0]}, {a[1]}), driver 1 at ({d[0]}, {d[1]}): '
          f'{across} across + {up} up = {blocks(a, d)} blocks = '
          f'{blocks(a, d) * BLOCK_SECONDS} s')
    cost = cost_table(riders, drivers)
    print(f'{len(riders)} riders and {len(drivers)} drivers in a '
          f'{NEIGHBOURHOOD} x {NEIGHBOURHOOD} neighbourhood; '
          f'blocks from each driver:')
    print_table(cost)
    nearest = cost.min(axis=1)
    wanted = [int(np.argmin(row)) for row in cost]
    print('nearest driver for each rider: ' + '  '.join(
        f'{RIDERS[r]}->{wanted[r] + 1} ({nearest[r]})'
        for r in range(len(riders))))
    clash = sorted({d + 1 for d in wanted if wanted.count(d) > 1})
    print(f'if everyone could have their nearest: {nearest.sum()} blocks = '
          f'{nearest.sum() * BLOCK_SECONDS} s in total. But drivers '
          f'{clash} are each wanted by more than one rider.')
    return {'riders': riders, 'drivers': drivers, 'cost': cost,
            'wanted': wanted, 'clash': clash, 'dream': int(nearest.sum())}


if __name__ == '__main__':
    run()
