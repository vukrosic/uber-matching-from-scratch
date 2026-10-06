"""Step 5. A whole evening: riders keep asking, trips take time, drivers
come back.

100 drivers, 18:00 to 21:00. About 5 riders ask per minute, and around 19:15
there is a rush of 12 per minute. A matched driver drives to the rider, then
to where the rider wants to go, and is free again there.

The dispatcher uses the obvious rule of step 2, every second: riders in the
order they asked, each gets the nearest free driver. No free driver: the
rider waits in line.
A rider's wait = seconds until matched + seconds until the car arrives.
"""
import numpy as np

from city import BLOCK_SECONDS, clock, cost_table, random_points
from step2_nearest_driver import nearest_driver

DRIVERS = 100
EVENING = 3 * 3600                        # seconds from 18:00 to 21:00
QUIET, RUSH = 5, 12                       # riders asking per minute
PEAK_AT, PEAK_WIDTH = 75 * 60, 25 * 60    # the rush: around 19:15


def demand(t, busy=1.0):
    """Riders asking per minute at second t of the evening."""
    bump = np.exp(-((t - PEAK_AT) / PEAK_WIDTH) ** 2)
    return busy * (QUIET + (RUSH - QUIET) * bump)


def make_requests(seed=0, busy=1.0):
    """One row per rider: the second they ask, where they are, where to."""
    rng = np.random.default_rng(seed)
    seconds = np.arange(EVENING)
    asking = rng.poisson(demand(seconds, busy) / 60)   # riders each second
    when = np.repeat(seconds, asking)
    start = random_points(len(when), rng)
    end = random_points(len(when), rng)
    return np.column_stack([when, start, end])         # t, x, y, to_x, to_y


def first_come(cost, waited):
    """The rule of step 2. The rows of cost are in the order riders asked."""
    return nearest_driver(cost)


def simulate(requests, rule, window=1, seed=0, road=cost_table, seen=None):
    """Every `window` seconds the dispatcher pairs waiting riders with free
    drivers. road = the true blocks between corners, seen = the blocks the
    dispatcher believes. Returns one row per rider (asked, matched, pickup
    seconds, driver) and, per minute, (second, riders waiting, drivers free).
    """
    seen = seen or road
    at = random_points(DRIVERS, np.random.default_rng(1000 + seed))
    free_from = np.zeros(DRIVERS)     # the second each driver is free again
    log = np.zeros((len(requests), 4), dtype=int)
    waiting, arrived, minutes, t = [], 0, [], 0
    while arrived < len(requests) or waiting:
        while arrived < len(requests) and requests[arrived, 0] <= t:
            waiting.append(arrived)   # a new rider joins the back of the line
            arrived += 1
        free = np.flatnonzero(free_from <= t)
        if t % 60 == 0:
            minutes.append((t, len(waiting), len(free)))
        if waiting and len(free) and t % window == 0:
            spots = requests[waiting, 1:3]
            waited = t - requests[waiting, 0]
            pairs = rule(seen(spots, at[free]), waited)
            drive = road(spots, at[free])
            for row, col in pairs:
                rider, driver = waiting[row], free[col]
                goal = requests[rider, 3:5]
                pickup = drive[row, col] * BLOCK_SECONDS
                trip = road(spots[row][None], goal[None])[0, 0]
                log[rider] = requests[rider, 0], t, pickup, driver
                free_from[driver] = t + pickup + trip * BLOCK_SECONDS
                at[driver] = goal     # where this driver is free next
            matched = {waiting[row] for row, _ in pairs}
            waiting = [r for r in waiting if r not in matched]
        t += 1
    return log, minutes


def waits(log):
    """Seconds from asking until the car arrives, for every rider."""
    return log[:, 1] - log[:, 0] + log[:, 2]


def by_time(log, quarter=900):
    """Average wait of the riders who asked in each quarter of an hour."""
    slot, w = log[:, 0] // quarter, waits(log)
    return [float(w[slot == s].mean()) for s in range(EVENING // quarter)]


def run():
    requests = make_requests()
    print(f'{len(requests):,} riders ask during the evening; per minute: '
          f'{demand(0):.1f} at 18:00, {demand(PEAK_AT):.1f} at 19:15, '
          f'{demand(EVENING):.1f} at 21:00')
    log, minutes = simulate(requests, first_come)
    w, curve, mins = waits(log), by_time(log), np.array(minutes)
    for s, v in enumerate(curve):
        line, free = mins[s * 15:s * 15 + 15, 1:].mean(axis=0)
        print(f'  asked {clock(s * 900)}-{clock(s * 900 + 900)}: wait '
              f'{v:5.0f} s   free drivers {free:5.1f}   riders in line '
              f'{line:5.1f}')
    calm = log[:, 0] < 1800
    rush = (log[:, 0] >= 3600) & (log[:, 0] < 5400)
    print(f'drive to the pickup: {log[calm, 2].mean() / BLOCK_SECONDS:.1f} '
          f'blocks for riders who asked before 18:30, '
          f'{log[rush, 2].mean() / BLOCK_SECONDS:.1f} blocks for riders who '
          f'asked 19:00-19:30')
    print(f'the evening: average wait {w.mean():.0f} s, 1 rider in 20 waits '
          f'over {np.percentile(w, 95):.0f} s, the worst {w.max():.0f} s')
    return {'requests': requests, 'log': log, 'minutes': minutes,
            'curve': curve}


if __name__ == '__main__':
    run()
