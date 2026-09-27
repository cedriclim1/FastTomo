# Task: harden the perp-exact SIRT discriminating test (review fix, test-only)

Adversarial review of the SIRT harness returned APPROVE-WITH-FIXES. Correctness is
confirmed; one MEDIUM test-adequacy gap remains. Fix ONLY the test — do not change any
harness/runner numerics. Do not commit.

## The gap (confirmed by the reviewer)

`tests/test_perrow_sirt.py:49` builds per-column angle offsets as `linspace(-3, 3, 11)`
(±3°). At that magnitude, perp-exact, mid-angle, and even column-REVERSED angle maps all
score correlation ~0.985 — indistinguishable. So the test that is supposed to prove the
perp-exact path actually consumes the exact per-column angles would stay green even if a
future refactor silently collapsed it to mid-angle or transposed the column↔angle map.

## Fix

In `tests/test_perrow_sirt.py`:
1. Raise the intra-frame per-column angle sweep to a realistic ±7–10° span (matching the
   real datasets' ~15°/frame), so the exact-vs-approx difference is resolvable.
2. Add/strengthen an assertion that perp-exact STRICTLY beats a mid-angle reconstruction
   on the SAME fast-perp synthetic data by a margin: `corr_exact > corr_mid + margin`
   (reviewer measured ~0.87 vs ~0.70 at this magnitude — pick a conservative margin like
   0.05 that holds with headroom, but verify empirically and set it from what you
   measure, not blindly). This locks in that the exact path uses θ(frame,slice)
   alignment, not just correct slab assembly.
3. Keep the existing constant-angle slab-equivalence test (:74-99) as-is — it's the
   tight assembly proof.

Do not weaken any existing assertion. The whole file must still pass.

## Validation

`conda run -n FastTomo-adrecon python -m pytest tests/test_perrow_sirt.py -v` (all pass),
then the full CPU suite, then `ruff check tests/test_perrow_sirt.py`. Append the new
pytest output and the measured corr_exact / corr_mid values to HANDOFF_ADRECON.md. LAST
action: write SIRT_TESTFIX_DONE.md at the worktree root with the measured margin.
