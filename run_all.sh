#!/bin/sh
# Runs every step in order and prints what the video shows. About a minute.
set -e
cd "$(dirname "$0")"
for f in step1_city step2_nearest_driver step3_every_pairing step4_auction \
         step5_evening step6_batching step7_which_part step8_fair \
         exercise_river; do
  echo "== $f"
  python3 $f.py
done
