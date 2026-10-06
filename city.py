"""The city every step shares: a grid of streets, with riders and drivers
standing on the corners."""
import numpy as np

SIZE = 40             # the city is 40 blocks wide and 40 blocks tall
BLOCK_SECONDS = 15    # a car needs 15 seconds to drive one block


def random_points(n, rng, size=SIZE):
    """n random street corners. Each one is a pair (x, y)."""
    return rng.integers(0, size + 1, (n, 2))


def blocks(a, b):
    """Street distance between two corners: blocks across + blocks up."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def cost_table(riders, drivers):
    """One row per rider, one column per driver: the blocks between them.
    Same as calling blocks() for every pair, done by numpy in one go."""
    riders, drivers = np.asarray(riders), np.asarray(drivers)
    return np.abs(riders[:, None, :] - drivers[None, :, :]).sum(axis=2)


def total(cost, pairs):
    """All blocks driven to pick everyone up. pairs = [(rider, driver)]."""
    return int(sum(cost[r][d] for r, d in pairs))


def clock(t, start_hour=18):
    """Second 4500 of an evening that starts at 18:00 -> '19:15'."""
    return f'{start_hour + int(t) // 3600}:{int(t) % 3600 // 60:02d}'


def how_long(seconds):
    """A length of time in words: 7200 -> '2.0 hours'."""
    units = (('years', 365 * 86400), ('days', 86400), ('hours', 3600),
             ('minutes', 60), ('seconds', 1))
    name, size = next(u for u in units if seconds >= u[1] or u[1] == 1)
    value = seconds / size
    if value >= 1e6:
        return f'{value:.1e} {name}'
    return f'{value:,.0f} {name}' if value >= 10 else f'{value:.1f} {name}'
