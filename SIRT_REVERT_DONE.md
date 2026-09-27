# SIRT revert complete

The SIRT harness now has exactly two reconstruction modes: `perp-exact` for
slice-wise standard `run_sirt` on fast-perp data, and `midangle` for standard
full-volume `run_sirt` with one per-frame angle. `auto` selects `perp-exact`
for fast-perp and `midangle` for fast-par. The exact-geometry implementation
and CLI option are removed, and `tests/test_perrow_sirt_exact.py` is absent.

Focused pytest tail:

```text
....                                                                     [100%]
4 passed in 0.86s
```

Full CPU-suite pytest tail:

```text
..........................................................s............. [ 54%]
................sssss....................................s.              [100%]
124 passed, 7 skipped in 7.56s
```

Ruff check:

```text
All checks passed!
```
