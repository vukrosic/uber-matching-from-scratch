# Ride matching from scratch

Which driver should pick up which rider? This repo builds the answer step by
step in plain Python and numpy: a city on a grid, the obvious rule, the best
possible pairing (first by brute force, then by an auction written from
scratch), a simulated evening with a rush hour, and batched matching.

Every number below is printed by the code in this folder. Seeds are fixed, so
your run prints the same numbers; only the timings in milliseconds depend on
your computer.

## Run it

```sh
bash run_all.sh        # every step in order, about a minute
python3 step4_auction.py   # or one step at a time
python3 tests.py       # checks the claims on evenings the steps never used
python3 values.py      # writes values.json, the numbers the video draws
```

Needs Python 3 and numpy. `demo-output.txt` is the saved output of
`run_all.sh`.

## The steps

| File | What it builds | What the run shows |
|---|---|---|
| `city.py` | The shared city: 40 x 40 blocks, 15 seconds per block, street distance, the cost table | used by every step |
| `step1_city.py` | Riders, drivers and the table of blocks between them | 6 riders, 6 drivers. If each rider could have their nearest driver: 16 blocks = 240 s. Drivers 1 and 2 are each wanted by three riders |
| `step2_nearest_driver.py` | The obvious rule: in the order riders asked, each gets the nearest free driver | 29 blocks = 435 s. Over 1,000 random moments the sixth rider waits 116 s, the first 46 s (2.5 times) |
| `step3_every_pairing.py` | A two-rider trap, then brute force over every pairing | Trap: 120 s against 60 s. Six riders: best of 720 pairings is 23 blocks, 90 s shorter. The obvious rule misses the best pairing in 68.9% of 1,000 moments and is 14.2% longer on average. 10 riders = 3,628,800 pairings; 20 riders would take tens of thousands of years |
| `step4_auction.py` | The auction: riders bid, prices rise, bumped riders bid again | Trap solved in 3 bids, six riders in 10 bids (23 blocks). Equal to brute force on 600 of 600 random tables up to 7 x 7. 100 riders and 100 drivers: 77.6 s per rider against 108.3 s (28% shorter), about 52,000 bids, under 2 seconds |
| `step5_evening.py` | A simulated evening: 100 drivers, 1,212 riders, a rush at 19:15, drivers busy until the trip ends | Obvious rule: waits go from 44 s to 1,003 s in the rush. Pickup drives grow from 3.4 to 21.7 blocks, which keeps drivers busy even longer. Evening average 427 s, worst 1,534 s |
| `step6_batching.py` | Batching: every 10 seconds, match everyone waiting with every free driver | Same evening: 125 s against 427 s (3.4 times shorter). Over 8 evenings: 122 s against 442 s. Calm evening: batching costs 4 s (67 s against 63 s). Window length 1 to 30 s changes the rush result by less than 5 s |
| `step7_which_part.py` | A check: closest pair first, a rule with no order of asking and no auction | On frozen 50 x 50 tables it is 21% longer than the auction. Over 24 rush evenings it gives 124.4 s, the auction 124.2 s; the auction wins on 14 of 24. Of 729 batches, the auction found a shorter total in 9 |
| `step8_fair.py` | Fairness: every 60 s a rider has waited takes one block off their row | Busier evenings: one rider waited 1,815 s (30 minutes), passed over in 151 batches. With waiting counted: 590 s. Over 24 evenings the worst wait falls 40% (1,539 s to 925 s), shorter on 24 of 24, while the average grows 1.8% (143 s to 146 s) |
| `exercise_river.py` | The exercise below, solved | see below |

Code size, without blank lines, comments and docstrings: the eight steps are
490 lines, plus 22 in `city.py`.

## What the measurements say

1. **Order of asking is the expensive part.** Serving riders first come,
   first served works while drivers are spare (63 s on a calm evening). When
   riders outnumber free drivers, each newly free driver is sent to whoever
   has waited longest, wherever they are. The pickup drive grows from 3.4 to
   21.7 blocks, drivers are tied up longer, and the line grows: 442 s average
   over 8 rush evenings.
2. **Batching fixes that**: 122 s on the same 8 evenings, better on 8 of 8.
   It pays only when the city is busy. At 60% and 70% of the riders it loses
   by about 4 seconds, roughly half the 10 second window.
3. **The exact best pairing matters less than expected in the running
   city.** On one frozen table the auction beats every simple rule, by 21%
   to 30% at 50 riders. In the simulated evening most batches have one rider
   or one driver on the short side (408 of 729), so a simple
   closest-pair-first rule reaches the same average wait within 1%.
4. **The table is where the policy lives.** The auction never changes after
   step 4. Fairness (step 8) and the river (exercise) are both changes to
   the numbers in the table.

## Exercise

A river runs down the middle of the city, between x = 20 and x = 21, with
bridges on the streets y = 10 and y = 30. Cars can only cross on a bridge.

1. Write `river_table(riders, drivers)`: the real blocks by road. Same side
   of the river: as before. Opposite sides: blocks across, plus the smallest
   detour up or down to a bridge.
2. `simulate` takes `road` (how far cars really drive) and `seen` (what the
   dispatcher believes). Run the 10 second batch on 24 evenings at
   `busy = 0.7` twice: once with `seen=cost_table` (blind to the river) and
   once with the river table. How much shorter is the average wait?

Measured answer (`python3 exercise_river.py`): a rider at (20, 20) with a
driver at (21, 20) looks 1 block away and is 21 blocks away by road. The
blind dispatcher averages 83.9 s, the one with the real table 79.1 s: 6%
shorter, better on 24 of 24 evenings. One wait in a hundred is over 376 s
blind and over 312 s with the real table. With a single bridge at y = 20 the
gap grows to 13% (106.3 s to 92.2 s); with three bridges it shrinks to 4%.
`tests.py` checks `river_table` against a walk through the street grid.

## What is simplified

- Riders and destinations are spread evenly over the city; every block takes
  15 seconds; there is no traffic.
- A driver takes one rider, never declines, and stays where the last trip
  ended. A rider never cancels. A match is never changed after it is made.
- The number of drivers is fixed at 100 and prices do not exist. Real
  dispatch systems also move drivers ahead of demand and change prices.
- The auction is the simple forward auction with one fixed smallest raise.
  It is exact for whole-number costs. Faster variants exist.

This is the textbook form of the matching problem. It makes no claim about
how any company's production system works.

## Licence

MIT for the code.
