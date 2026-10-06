"""Step 7. Which part of step 6 did the work?

Step 6 changed two things at once: the order of asking was dropped, and the
auction found the best pairing. To tell them apart, a third rule drops the
order of asking and has no auction: closest pair first.

On a frozen table it is clearly worse than the auction. In the running city
it is not, and the batches themselves show why.
"""
import numpy as np

from city import cost_table, random_points, total
from step1_city import snapshot
from step2_nearest_driver import nearest_driver
from step4_auction import best_match
from step5_evening import first_come, make_requests, simulate
from step6_batching import average, batch

EVENINGS = 24
RULES = (('obvious rule, in order of asking', first_come, 1),
         ('closest pair first, every second', None, 1),
         ('closest pair first, every 10 s', None, 10),
         ('auction, every second', batch, 1),
         ('auction, every 10 s', batch, 10))


def closest_pair_first(cost, waited=None):
    """No order of asking and no auction: the closest rider-driver pair
    goes first, then the closest pair among those left, and so on."""
    cost, pairs = np.array(cost, dtype=float), []
    for _ in range(min(cost.shape)):
        rider, driver = np.unravel_index(np.argmin(cost), cost.shape)
        pairs.append((int(rider), int(driver)))
        cost[rider, :] = cost[:, driver] = np.inf
    return sorted(pairs)


def run():
    six = cost_table(*snapshot())
    print(f'the six riders: nearest driver {total(six, nearest_driver(six))}'
          f' blocks, closest pair first '
          f'{total(six, closest_pair_first(six))} blocks, auction '
          f'{total(six, best_match(six))} blocks')
    rng, frozen = np.random.default_rng(0), {}
    for n in (6, 20, 50):             # frozen moments, riders = drivers
        sums = np.zeros(3)
        for _ in range(20):
            c = cost_table(random_points(n, rng), random_points(n, rng))
            sums += [total(c, rule(c)) for rule in
                     (nearest_driver, closest_pair_first, best_match)]
        frozen[n] = (sums[:2] / sums[2] - 1).tolist()
        print(f'  20 frozen moments, {n:>2} riders: nearest driver '
              f'{frozen[n][0]:4.0%} longer than the auction, closest pair '
              f'first {frozen[n][1]:4.0%} longer')

    print(f'{EVENINGS} rush evenings: average wait, 1 in 20 waits over')
    rows = {}
    for name, rule, window in RULES:
        r = rows[name] = average(rule or closest_pair_first, window,
                                 seeds=EVENINGS)
        print(f'  {name:<34} {r["wait"]:5.1f} s  {r["p95"]:5.0f} s')
    cheap, exact = (rows[RULES[i][0]]['each'] for i in (2, 4))
    wins = sum(y < x for x, y in zip(cheap, exact))
    print(f'  every 10 s, the auction beats closest pair first on {wins} '
          f'of {EVENINGS} evenings')

    sizes = []

    def spy(cost, waited):            # the batch rule, taking notes
        pairs = best_match(cost)
        other = total(cost, closest_pair_first(cost))
        sizes.append((*cost.shape, other - total(cost, pairs)))
        return pairs
    simulate(make_requests(), spy, 10)
    sizes = np.array(sizes)
    small = sizes[:, :2].min(axis=1)  # the short side of each table
    print(f'why: the first evening has {len(sizes)} batches of 10 s. '
          f'{(small == 1).sum()} have one rider or one driver, where both '
          f'rules must agree; {(small == 2).sum()} have two on the short '
          f'side; {(small > 2).sum()} have more.')
    print(f'  the auction found a shorter total than closest pair first in '
          f'{(sizes[:, 2] > 0).sum()} batches, {sizes[:, 2].sum()} blocks '
          f'in the whole evening')
    return {'frozen': frozen, 'rows': rows, 'wins': wins, 'sizes': sizes}


if __name__ == '__main__':
    run()
