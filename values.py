"""Runs every step once more and saves the numbers, positions and curves the
film draws: values.json."""
import contextlib
import glob
import io
import json
import tokenize
from math import factorial

import numpy as np

import city
import exercise_river as ex
import step1_city as s1
import step2_nearest_driver as s2
import step3_every_pairing as s3
import step4_auction as s4
import step5_evening as s5
import step6_batching as s6
import step7_which_part as s7
import step8_fair as s8

R = lambda a, k=2: np.round(np.asarray(a, dtype=float), k).tolist()
BS = city.BLOCK_SECONDS


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def pairs_of(cost, pairs):
    """[[rider, driver, blocks], ...] for one pairing, and its total."""
    rows = [[int(r), int(d), int(cost[r][d])] for r, d in pairs]
    return {'pairs': rows, 'blocks': sum(p[2] for p in rows),
            'seconds': sum(p[2] for p in rows) * BS}


def bids(log):
    """An auction, bid by bid: bills before the bid, prices after it."""
    return [{'rider': b['rider'], 'driver': b['driver'],
             'bumped': b['bumped'], 'bill': R(b['bill'], 3),
             'runner_up': round(b['runner_up'], 3),
             'raise': round(b['raise'], 3), 'price': R(b['price'], 3)}
            for b in log]


def trips(requests, log, seed=0):
    """One row per rider for drawing an evening: asked, x, y, to_x, to_y,
    matched, pickup seconds, driver, and where that driver came from."""
    at = city.random_points(s5.DRIVERS, np.random.default_rng(1000 + seed))
    at, rows = at.copy(), np.zeros((len(log), 10), dtype=int)
    for i in np.argsort(log[:, 1], kind='stable'):
        d = log[i, 3]
        rows[i] = *requests[i], log[i, 1], log[i, 2], d, *at[d]
        assert city.blocks(at[d], requests[i, 1:3]) * BS == log[i, 2]
        at[d] = requests[i, 3:5]
    return rows.tolist()


def summary(log):
    w = s5.waits(log)
    return {'mean': round(float(w.mean()), 1), 'worst': int(w.max()),
            'p95': round(float(np.percentile(w, 95)), 1),
            'pickup_blocks': round(float(log[:, 2].mean()) / BS, 2),
            'over10': int((w > 600).sum())}


def avg(r):
    """One row of step 6's `average`, rounded."""
    return {k: (R(v, 1) if isinstance(v, list) else round(v, 1))
            for k, v in r.items()}


def code_lines(path):
    """Lines of code: no blank lines, comments or docstrings."""
    keep, skip = set(), (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                         tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)
    for tok in tokenize.generate_tokens(open(path).readline):
        doc = tok.type == tokenize.STRING and tok.string.startswith('"""')
        if tok.type not in skip and not doc:
            keep.update(range(tok.start[0], tok.end[0] + 1))
    return len(keep)


def main():
    V = {'city': {'size': city.SIZE, 'block_seconds': BS,
                  'neighbourhood': s1.NEIGHBOURHOOD, 'drivers': s5.DRIVERS,
                  'evening_seconds': s5.EVENING, 'quiet': s5.QUIET,
                  'rush': s5.RUSH, 'peak_at': s5.PEAK_AT,
                  'peak_width': s5.PEAK_WIDTH, 'names': s1.RIDERS}}

    # steps 1 and 2: the six riders and six drivers that are pictured
    r1, r2 = quiet(s1.run), quiet(s2.run)
    cost = r1['cost']
    V['six'] = {'riders': r1['riders'], 'drivers': r1['drivers'],
                'cost': cost, 'wanted': r1['wanted'], 'clash': r1['clash'],
                'dream_blocks': r1['dream'],
                'dream_seconds': r1['dream'] * BS,
                'nearest': pairs_of(cost, r2['pairs']),
                'closest_pair_first': pairs_of(
                    cost, s7.closest_pair_first(cost)),
                'best': pairs_of(cost, s3.every_pairing(cost)[0]),
                'by_turn_seconds': R(r2['by_turn'], 1)}

    # step 3: the trap, the 1,000 moments, the explosion
    r3 = quiet(s3.run)
    trap = r3['trap']
    V['trap'] = {'riders': s3.TRAP_RIDERS, 'drivers': s3.TRAP_DRIVERS,
                 'cost': trap,
                 'nearest': pairs_of(trap, s2.nearest_driver(trap)),
                 'best': pairs_of(trap, s3.every_pairing(trap)[0])}
    V['brute'] = {'missed_percent': r3['worse'] / 10, 'beaten': r3['better'],
                  'obvious_seconds': round(r3['obvious'] * BS, 1),
                  'best_seconds': round(r3['best'] * BS, 1),
                  'longer': round(r3['obvious'] / r3['best'] - 1, 4),
                  'timed': [{'n': n, 'pairings': k, 'ms': round(t * 1000, 2)}
                            for n, k, t in r3['rows']],
                  'per_second': round(r3['speed']),
                  'estimates': [{'n': n, 'pairings': float(factorial(n)),
                                 'seconds': s, 'text': city.how_long(s)}
                                for n, s in r3['estimates']]}

    # step 4: two auctions bid by bid, the check, and the big tables
    r4 = quiet(s4.run)
    V['auction'] = {'trap': bids(r4['trap_log']), 'six': bids(r4['log6']),
                    'trap_step': 1 / 3, 'six_step': 1 / 7,
                    'six_result': pairs_of(cost, r4['pairs6']),
                    'equal': r4['same'], 'cases': r4['cases'],
                    'sizes': [{k: round(v, 2) for k, v in row.items()}
                              for row in r4['rows']]}

    # steps 5 and 6: the first evening under both rules, then the sweeps
    r6 = quiet(s6.run)
    req, old, new = r6['requests'], r6['old'], r6['new']
    mins = {'obvious': quiet(s5.simulate, req, s5.first_come)[1],
            'batch': quiet(s5.simulate, req, s6.batch, 10)[1]}
    calm = old[:, 0] < 1800
    rush = (old[:, 0] >= 3600) & (old[:, 0] < 5400)
    V['evening'] = {
        'riders': len(req),
        'demand_per_minute': R(s5.demand(np.arange(0, s5.EVENING + 1, 60))),
        'drivers_start': city.random_points(
            s5.DRIVERS, np.random.default_rng(1000)),
        'columns': ['asked', 'x', 'y', 'to_x', 'to_y', 'matched',
                    'pickup_seconds', 'driver', 'driver_x', 'driver_y'],
        'obvious': {'trips': trips(req, old), **summary(old),
                    'by_quarter': R(s5.by_time(old), 1),
                    'minutes': mins['obvious'],
                    'pickup_blocks_calm': round(old[calm, 2].mean() / BS, 2),
                    'pickup_blocks_rush': round(old[rush, 2].mean() / BS, 2)},
        'batch': {'trips': trips(req, new), **summary(new), 'window': 10,
                  'by_quarter': R(s5.by_time(new), 1),
                  'minutes': mins['batch']}}
    V['windows'] = {
        name: {'obvious': avg(s6.average(s5.first_come, 1, busy)),
               'batch': [{'window': w, **avg(r6['windows'][busy][w])}
                         for w in s6.WINDOWS]}
        for name, busy in (('calm', s6.CALM), ('rush', 1.0))}
    V['busy'] = [{'busy': busy, 'obvious': avg(a), 'batch': avg(b),
                  'batch_better_on': sum(y < x for x, y in
                                         zip(a['each'], b['each']))}
                 for busy, (a, b) in r6['busy'].items()]

    # step 7: which part helps
    r7 = quiet(s7.run)
    small = r7['sizes'][:, :2].min(axis=1)
    V['which_part'] = {
        'frozen': [{'n': n, 'nearest_longer': round(a, 4),
                    'closest_pair_longer': round(b, 4)}
                   for n, (a, b) in r7['frozen'].items()],
        'evenings': s7.EVENINGS, 'auction_wins': r7['wins'],
        'rules': [{'name': k, **avg(v)} for k, v in r7['rows'].items()],
        'batches': {'count': len(small),
                    'short_side_1': int((small == 1).sum()),
                    'short_side_2': int((small == 2).sum()),
                    'short_side_more': int((small > 2).sum()),
                    'auction_shorter_in': int((r7['sizes'][:, 2] > 0).sum()),
                    'blocks_saved': int(r7['sizes'][:, 2].sum()),
                    'tables': r7['sizes']}}   # riders, drivers, blocks saved

    # step 8: fairness
    r8 = quiet(s8.run)
    who, w, w2 = r8['who'], s5.waits(r8['plain']), s5.waits(r8['fair'])
    edges = np.arange(0, 1921, 60)
    V['fair'] = {
        'busy': s8.BUSIER, 'window': s8.WINDOW, 'worth': s8.WORTH,
        'evenings': s8.EVENINGS,
        'rider': {'asked': r8['requests'][who, 0],
                  'x': r8['requests'][who, 1],
                  'y': r8['requests'][who, 2], 'wait_plain': w[who],
                  'wait_fair': w2[who], 'passed_by': r8['passed'],
                  'batches': (r8['plain'][who, 1] - r8['plain'][who, 0])
                  // s8.WINDOW},
        'plain': summary(r8['plain']), 'fair': summary(r8['fair']),
        'histogram_edges': edges,
        'histogram_plain': np.histogram(w, edges)[0],
        'histogram_fair': np.histogram(w2, edges)[0],
        'sweep': [{'worth': k, **avg(v)} for k, v in r8['rows'].items()],
        'rush': {'plain': avg(r8['rush'][0]), 'fair': avg(r8['rush'][1])}}

    # the exercise: a river
    rx = quiet(ex.run)
    rider, drivers, plain, real = rx['example']
    V['river'] = {'x': ex.RIVER_X, 'bridges': ex.BRIDGES, 'busy': ex.BUSY,
                  'evenings': ex.EVENINGS,
                  'layouts': {k: v for k, v in ex.LAYOUTS.items()},
                  'example': {'rider': rider[0], 'drivers': drivers,
                              'map_blocks': plain, 'road_blocks': real},
                  'rows': [{'name': k, 'blind': avg(a), 'real': avg(b),
                            'real_better_on': sum(y < x for x, y in
                                                  zip(a['each'], b['each']))}
                           for k, (a, b) in rx['rows'].items()]}

    files = sorted(glob.glob('step*.py')) + ['city.py', 'exercise_river.py']
    V['code'] = {'files': {p: {'lines': len(open(p).read().splitlines()),
                               'code_lines': code_lines(p)} for p in files}}
    V['code']['steps'] = sum(v['code_lines'] for p, v in
                             V['code']['files'].items() if p[:4] == 'step')
    V['code']['all'] = sum(v['code_lines']
                           for v in V['code']['files'].values())

    text = json.dumps(V, default=lambda o: o.tolist())
    open('values.json', 'w').write(text)
    print('values.json written,', len(text) // 1000, 'kB')


if __name__ == '__main__':
    main()
