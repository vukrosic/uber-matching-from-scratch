"""Step 6. Batching: wait a few seconds, then match everyone together.

Instead of answering each rider the second they ask, the dispatcher wakes up
every `window` seconds, takes every waiting rider and every free driver, and
runs the auction of step 4 on that table. The order of asking is dropped.

Measured here, on the same evenings for every rule:
  1. the evening of step 5, obvious rule against a 10 second batch;
  2. how long the window should be, on a calm and on the rush evening;
  3. how busy the city has to be before batching pays.
"""
from functools import lru_cache

import numpy as np

from city import clock, cost_table
from step4_auction import best_match
from step5_evening import by_time, first_come, make_requests, simulate, waits

WINDOWS = (1, 5, 10, 20, 30, 60, 120)
BUSY = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)
CALM = 0.6                            # an evening with 60% of the riders
NAMES = ('wait', 'matching', 'pickup', 'p95', 'p99', 'worst', 'over10')


def batch(cost, waited):
    """Everyone waiting and every free driver, matched by the auction."""
    return best_match(cost)


@lru_cache(maxsize=None)
def average(rule, window, busy=1.0, road=cost_table, seen=None, seeds=8):
    """The same evenings (seeds 0, 1, 2, ...) under one rule. Averages, in
    seconds; over10 = riders per evening who waited more than 10 minutes."""
    rows = []
    for seed in range(seeds):
        requests = make_requests(seed, busy)
        log, _ = simulate(requests, rule, window, seed, road, seen)
        w = waits(log)
        rows.append([w.mean(), (log[:, 1] - log[:, 0]).mean(),
                     log[:, 2].mean(), np.percentile(w, 95),
                     np.percentile(w, 99), w.max(), (w > 600).sum()])
    out = dict(zip(NAMES, np.mean(rows, axis=0).tolist()))
    out['each'] = [float(r[0]) for r in rows]     # average of each evening
    out['worst_each'] = [int(r[5]) for r in rows]
    return out


def run():
    requests = make_requests()
    old = simulate(requests, first_come)[0]
    new = simulate(requests, batch, 10)[0]
    print('the evening of step 5, average wait by the time riders asked:')
    for s, (a, b) in enumerate(zip(by_time(old), by_time(new))):
        print(f'  {clock(s * 900)}-{clock(s * 900 + 900)}: obvious rule '
              f'{a:5.0f} s   batch {b:4.0f} s')
    a, b = waits(old), waits(new)
    print(f'  whole evening: obvious rule {a.mean():.0f} s (worst {a.max()} '
          f's), 10 s batch {b.mean():.0f} s (worst {b.max()} s): '
          f'{a.mean() / b.mean():.1f} times shorter')

    print('window length, 8 evenings: wait = until matched + pickup drive')
    for name, busy in (('calm evening', CALM), ('rush evening', 1.0)):
        base = average(first_come, 1, busy)
        print(f'  {name}, obvious rule: {base["wait"]:.0f} s')
        for w in WINDOWS:
            r = average(batch, w, busy)
            print(f'    window {w:>3} s: {r["wait"]:5.1f} = '
                  f'{r["matching"]:5.1f} + {r["pickup"]:5.1f}   '
                  f'1 in 20 waits over {r["p95"]:.0f} s')

    print('how busy (1.0 = the rush evening): obvious rule / 10 s batch')
    for busy in BUSY:
        a, b = average(first_come, 1, busy), average(batch, 10, busy)
        wins = sum(y < x for x, y in zip(a['each'], b['each']))
        print(f'  busy {busy:.1f}: {a["wait"]:4.0f} s / {b["wait"]:4.0f} s'
              f'   batch is better on {wins} of 8 evenings')
    return {'old': old, 'new': new, 'requests': requests,
            'windows': {busy: {w: average(batch, w, busy) for w in WINDOWS}
                        for busy in (CALM, 1.0)},
            'busy': {busy: (average(first_come, 1, busy),
                            average(batch, 10, busy)) for busy in BUSY}}


if __name__ == '__main__':
    run()
