"""Step 4. The best pairing without trying them all: an auction.

Every driver has a price, in blocks. All prices start at 0. A rider without
a driver looks at each driver's bill = blocks away + price, takes the
cheapest, and raises that driver's price by as much as they can before the
runner-up would be the better deal. Whoever held that driver is bumped and
bids again. A driver that many riders want gets expensive, and all but one
of them move on.

The smallest raise is 1 / (riders + 1). The auction then ends less than one
block away from the best total. Block totals are whole numbers, so the
result is exactly the best.
"""
import time

import numpy as np

from city import BLOCK_SECONDS, cost_table, random_points, total
from step1_city import RIDERS, snapshot
from step2_nearest_driver import nearest_driver
from step3_every_pairing import TRAP_DRIVERS, TRAP_RIDERS, every_pairing


def auction(cost, log=None):
    """cost[rider][driver] = blocks. Needs riders <= drivers."""
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    step = 1 / (n + 1)                # the smallest raise
    price = np.zeros(m)               # every driver starts at price 0
    holder = [None] * m               # which rider holds each driver
    queue = list(range(n))            # riders who still need a driver
    while queue:
        rider = queue.pop(0)
        bill = cost[rider] + price    # what each driver costs this rider
        driver = int(np.argmin(bill))             # the cheapest one
        others = np.delete(bill, driver)
        runner_up = others.min() if m > 1 else bill[driver]
        raise_by = runner_up - bill[driver] + step
        price[driver] += raise_by
        bumped = holder[driver]
        if bumped is not None:
            queue.append(bumped)      # the old holder has to bid again
        holder[driver] = rider
        if log is not None:
            log.append({'rider': rider, 'bill': bill.tolist(),
                        'driver': driver, 'runner_up': float(runner_up),
                        'raise': float(raise_by), 'price': price.tolist(),
                        'bumped': bumped})
    return sorted((r, d) for d, r in enumerate(holder) if r is not None)


def best_match(cost):
    """The best pairs for any table. More riders than drivers? Turn the
    table on its side, so the drivers bid for riders, and turn it back."""
    cost = np.asarray(cost)
    if cost.shape[0] <= cost.shape[1]:
        return auction(cost)
    return sorted((r, d) for d, r in auction(cost.T))


def show(log):
    for i, b in enumerate(log):
        bumps = b['bumped'] is not None and f', bumps {RIDERS[b["bumped"]]}'
        print(f'  bid {i + 1:>2}: rider {RIDERS[b["rider"]]} sees bills '
              + ' '.join(f'{v:5.2f}' for v in b['bill'])
              + f' -> driver {b["driver"] + 1}, raises its price by '
              f'{b["raise"]:.2f}' + (bumps or ''))


def run():
    trap, trap_log = cost_table(TRAP_RIDERS, TRAP_DRIVERS), []
    trap_pairs = auction(trap, trap_log)
    print('the trap of step 3 (smallest raise 1/3 of a block):')
    show(trap_log)
    print(f'  result: {total(trap, trap_pairs)} blocks, the best of step 3')

    cost, log6 = cost_table(*snapshot()), []
    pairs6 = auction(cost, log6)
    print('the six riders (smallest raise 1/7 of a block):')
    show(log6)
    print('  result: ' + '  '.join(f'{RIDERS[r]}->{d + 1}' for r, d in pairs6)
          + f' = {total(cost, pairs6)} blocks after {len(log6)} bids; brute'
          f' force tried 720 pairings to find '
          f'{total(cost, every_pairing(cost)[0])} blocks')

    rng, same, cases = np.random.default_rng(0), 0, 600
    for _ in range(cases):            # any shape up to 7 x 7
        n, m = rng.integers(1, 8, 2)
        c = cost_table(random_points(n, rng, 10), random_points(m, rng, 10))
        truth = every_pairing(c if n <= m else c.T)[0]
        same += (total(c, best_match(c))
                 == total(c if n <= m else c.T, truth))
    print(f'auction equals brute force on {same} of {cases} random tables '
          f'up to 7 x 7')

    print('riders = drivers, whole city, average of 10 moments:')
    rows = []
    for n in (10, 20, 50, 100):
        near = best = bids = ms = 0
        for _ in range(10):
            c = cost_table(random_points(n, rng), random_points(n, rng))
            log, t0 = [], time.process_time()
            pairs = auction(c, log)
            ms += (time.process_time() - t0) * 100
            near += total(c, nearest_driver(c)) * BLOCK_SECONDS / n / 10
            best += total(c, pairs) * BLOCK_SECONDS / n / 10
            bids += len(log) / 10
        rows.append({'n': n, 'near': near, 'best': best, 'bids': bids,
                     'ms': ms})
        print(f'  {n:>3} riders: nearest driver {near:5.1f} s per rider, '
              f'auction {best:5.1f} s ({1 - best / near:.0%} shorter), '
              f'{bids:>7,.0f} bids, {ms:6.1f} ms')
    return {'trap_log': trap_log, 'trap_pairs': trap_pairs, 'log6': log6,
            'pairs6': pairs6, 'same': same, 'cases': cases, 'rows': rows}


if __name__ == '__main__':
    run()
