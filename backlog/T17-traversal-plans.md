# T17 · Traversal plans in the catalog

Tier: 2 · Owner: _ · Depends: T13

## What

Add top-down and multi-subtree traversal (fixed beam widths) as catalog strategies; measure the oracle gap with and without them on dev.

## Done when

A `NOTES.md` paragraph answers: do traversal plans reach any (tokens, F1) point that collapsed configurations cannot?

## Description

These plans were removed from the core because their benefit was search latency, which is negligible at our scale (proposal §5.1). This ticket only checks whether they add quality–cost points.
