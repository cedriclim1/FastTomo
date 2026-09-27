# Task: revert the SIRT harness to STANDARD SIRT only (remove exact-geometry mode)

An exact-geometry SIRT mode was added that changes SIRT's operator (per-ray forward).
That is NOT wanted — SIRT must stay standard single-axis Radon (one angle per
projection). Remove it entirely and restore the harness to exactly two modes:
`perp-exact` (slice-wise standard run_sirt) and `midangle` (standard run_sirt with the
per-frame angle). Do not commit.

## Do

1. `benchmarks/scripts/utils/perrow_sirt.py`: delete `sirt_exact_geometry`,
   `exact_geometry_adjoint`, the `exact-geometry` branch in `sirt_continuous`, and every
   `"exact-geometry"` from the `SIRTMode` Literal and validation lists. Restore
   `sirt_continuous` to dispatch only `perp-exact` / `midangle`. Remove now-unused
   imports (e.g. anything from perrow_ad only used by the deleted code) and docstring
   mentions of exact-geometry.
2. `benchmarks/scripts/recon_continuous_sirt.py`: remove `exact-geometry` from `--mode`
   choices and any handling; `auto` stays perp-exact (fast-perp) / midangle (fast-par).
3. Ensure `tests/test_perrow_sirt_exact.py` is gone (it was deleted; confirm).
4. Run `conda run -n FastTomo-adrecon python -m pytest tests/test_perrow_sirt.py -q`
   and the full CPU suite — all pass. `ruff check` the two files.
5. Remove the "SIRT exact-geometry" section from HANDOFF_ADRECON.md if present.

LAST action: write SIRT_REVERT_DONE.md at the worktree root confirming the two remaining
modes and pasting the pytest tail.
