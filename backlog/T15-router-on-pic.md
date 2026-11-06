# T15 · Router on the PIC index (Option C)

Tier: 2 · Owner: _ · Depends: T13

## What

Collect train labels on PIC-capped, retrain the router, recalibrate θ on dev, and run E1 on the PIC-capped test index (test labels already exist from T12).

## Done when

`results/test/h1_by_index.csv` shows H1 on F-100 and PIC-capped side by side, with CIs.

## Description

This tests whether the router's gains and the chunking gains add up. It costs about one extra train labeling pass.
