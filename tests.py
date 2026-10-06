"""Quick checks for every step. Run: python3 tests.py  (under a minute)

The steps use evenings with seeds 0 to 23. The evenings tested here use
seeds 24 to 29, so every claim is checked on evenings no step looked at.
"""
from collections import deque
from functools import partial

import numpy as np

from city import BLOCK_SECONDS, blocks, cost_table, random_points, total
from exercise_river import RIVER_X, river_table
from step2_nearest_driver import nearest_driver
from step3_every_pairing import TRAP_DRIVERS, TRAP_RIDERS, every_pairing
from step4_auction import auction, best_match
from step5_evening import DRIVERS, first_come, make_requests, simulate, waits
from step6_batching import batch
from step7_which_part import closest_pair_first
from step8_fair import fair

rng = np.random.default_rng(42)
SEEDS = range(24, 30)

# step 1: the table is blocks() for every pair
riders, drivers = random_points(5, rng), random_points(7, rng)
table = cost_table(riders, drivers)
assert table.shape == (5, 7)
assert all(table[r][d] == blocks(riders[r], drivers[d])
           for r in range(5) for d in range(7))

# step 2: no driver is given twice, and the first rider gets their nearest
for _ in range(50):
    c = cost_table(random_points(6, rng, 10), random_points(4, rng, 10))
    pairs = nearest_driver(c)
    assert len(pairs) == 4 and len({d for _, d in pairs}) == 4
    assert c[0][pairs[0][1]] == c[0].min()

# step 3: the trap, and brute force is never beaten by the obvious rule
trap = cost_table(TRAP_RIDERS, TRAP_DRIVERS)
assert total(trap, nearest_driver(trap)) == 8
assert total(trap, every_pairing(trap)[0]) == 4
assert every_pairing(np.zeros((4, 6)))[1] == 6 * 5 * 4 * 3

# step 4: the auction equals brute force on every shape up to 7 x 7
checked = 0
for n in range(1, 8):
    for m in range(1, 8):
        for _ in range(6):
            c = cost_table(random_points(n, rng, 12),
                           random_points(m, rng, 12))
            wide = c if n <= m else c.T
            best = total(wide, every_pairing(wide)[0])
            pairs = best_match(c)
            assert total(c, pairs) == best, (n, m)
            assert len(pairs) == min(n, m)
            assert len({r for r, _ in pairs}) == len({d for _, d in pairs})
            assert total(c, nearest_driver(c)) >= best
            checked += 1
assert checked == 294

# step 4: prices only go up, and on big tables the auction is never worse
log = []
auction(cost_table(random_points(8, rng), random_points(8, rng)), log)
prices = np.array([b['price'] for b in log])
assert (np.diff(prices, axis=0) >= 0).all() and (prices[0] >= 0).all()
for _ in range(20):
    c = cost_table(random_points(30, rng), random_points(30, rng))
    best = total(c, best_match(c))
    assert best <= total(c, nearest_driver(c))
    assert best <= total(c, closest_pair_first(c))


def sound(requests, log, road=cost_table):
    """Every rider is driven once, and no driver is in two places."""
    assert (log[:, 1] >= log[:, 0]).all() and (log[:, 2] >= 0).all()
    trip = np.array([road(r[None, 1:3], r[None, 3:5])[0, 0]
                     for r in requests])
    done = log[:, 1] + log[:, 2] + trip * BLOCK_SECONDS
    for d in range(DRIVERS):
        mine = np.flatnonzero(log[:, 3] == d)
        mine = mine[np.argsort(log[mine, 1], kind='stable')]
        assert (log[mine[1:], 1] >= done[mine[:-1]]).all()


# steps 5 to 7: the simulator is sound, and the claims hold on new evenings
gap, cheap = [], []
for seed in SEEDS:
    rush, calm = make_requests(seed), make_requests(seed, 0.6)
    old = simulate(rush, first_come, 1, seed)[0]
    new = simulate(rush, batch, 10, seed)[0]
    sound(rush, old), sound(rush, new)
    # in the rush, the 10 s batch at least halves the average wait
    assert waits(new).mean() < 0.5 * waits(old).mean(), seed
    # the order of asking is kept by the obvious rule, dropped by the batch
    assert (np.diff(old[:, 1]) >= 0).all() and (np.diff(new[:, 1]) < 0).any()
    # on a calm evening the batch only costs seconds: about half the window
    a = waits(simulate(calm, first_come, 1, seed)[0]).mean()
    b = waits(simulate(calm, batch, 10, seed)[0]).mean()
    gap.append(b - a)
    # the cheap rule with no auction gives the same average, within 5%
    c = waits(simulate(rush, closest_pair_first, 10, seed)[0]).mean()
    cheap.append(c / waits(new).mean())
assert 2 < np.mean(gap) < 8 and min(gap) > 0, gap
assert 0.95 < min(cheap) and max(cheap) < 1.05, cheap

# step 8: counting the wait cuts the worst wait and barely moves the average
plain, fairer = [], []
for seed in SEEDS:
    busier = make_requests(seed, 1.1)
    for rule, out in ((batch, plain), (fair, fairer)):
        log = simulate(busier, rule, 10, seed)[0]
        sound(busier, log)
        out.append((waits(log).mean(), waits(log).max()))
plain, fairer = np.mean(plain, axis=0), np.mean(fairer, axis=0)
assert fairer[1] < 0.85 * plain[1], (plain, fairer)
assert abs(fairer[0] / plain[0] - 1) < 0.05, (plain, fairer)


def walk(start, bridges, size=40):
    """Blocks from start to every corner, found by walking the streets."""
    far = {start: 0}
    todo = deque([start])
    while todo:
        x, y = todo.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            wet = {x, nx} == {RIVER_X, RIVER_X + 1} and y not in bridges
            inside = 0 <= nx <= size and 0 <= ny <= size
            if inside and not wet and (nx, ny) not in far:
                far[(nx, ny)] = far[(x, y)] + 1
                todo.append((nx, ny))
    return far


# exercise: the river table is the true shortest drive, checked by walking
for bridges in ((10, 30), (20,), (5, 20, 35)):
    for sx, sy in random_points(4, rng):
        far = walk((int(sx), int(sy)), bridges)
        ends = random_points(60, rng)
        got = river_table([(sx, sy)], ends, bridges)[0]
        assert all(got[i] == far[(int(x), int(y))]
                   for i, (x, y) in enumerate(ends))

# exercise: a dispatcher that sees the river gives shorter waits
for bridges in ((10, 30), (20,)):
    road, blind, sees = partial(river_table, bridges=bridges), [], []
    for seed in SEEDS:
        evening = make_requests(seed, 0.7)
        a = simulate(evening, batch, 10, seed, road, cost_table)[0]
        b = simulate(evening, batch, 10, seed, road)[0]
        sound(evening, a, road), sound(evening, b, road)
        blind.append(waits(a).mean()), sees.append(waits(b).mean())
    assert np.mean(sees) < np.mean(blind), (bridges, blind, sees)
    assert sum(s < b for s, b in zip(sees, blind)) >= 4, (blind, sees)

print('all tests pass')
