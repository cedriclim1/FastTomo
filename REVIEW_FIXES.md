# Review fixes for the perrow_ad harness (apply exactly these, nothing else)

Adversarial review verdict on your TASK_ADRECON.md deliverables: APPROVE-WITH-FIXES.
Geometry mapping and chunked-gradient math were confirmed correct. Apply the three
fixes below in this worktree. Do NOT change any numerics of the reconstruction path
(other than fix 3, which must be behavior-neutral). Do not commit.

## Fix 1 (main): even-size forward test + honest tolerance + doc note

The forward test uses only ODD volume sizes (21, 11). At odd sizes FastTomo's
ray-marched forward_project coincides with the generator's rotate-then-sum to ~1e-6,
but at EVEN sizes (production data is 180×180 and 200×200) they diverge by ~3-5% rel
at nonzero angles (angle-dependent integration-scheme difference, NOT a centering bug;
θ=0 matches at both parities).

- Add even-size parametrizations (e.g. size=20, both arms, both Δθ values) to
  tests/test_perrow_ad.py::test_forward_matches_absolute_generator_frame.
- For the even cases use a tolerance that reflects the true model gap (measure it,
  then assert e.g. relative max error < 0.06 AND assert it is NOT small (> 0.5%),
  so the gap is pinned honestly rather than hidden). Keep the tight 1e-4/1e-3
  tolerance for the odd cases (absolute-frame proof).
- Add a clear note to the perrow_ad.py module docstring AND a short section to
  HANDOFF_ADRECON.md: the harness forward operator is not generator-identical at
  even sizes; expected ~2-5% relative model mismatch on production 180/200-sized
  data; fidelity floor is raised accordingly (~1e-3 scale at recon time). This is
  accepted as inverse-crime avoidance, not a bug.

## Fix 2: non-square frame test

All test volumes are cubic, so an image_y↔image_x transposition in frame handling
would be undetectable on square data. Add one non-square case (e.g. generator volume
(z,y,x) = (12, 14, 10)) exercising the forward match for both arms at odd sizes with
tight tolerance.

## Fix 3: harden predict() gather buffer

perrow_ad.py predict() builds the expanded prediction with new_empty + index_copy.
Coverage is currently complete, but a future gap would silently return uninitialized
memory. Switch to new_zeros (document with a one-line comment why). Verify no test
output changes.

## Validation

Run: conda run -n FastTomo-adrecon python -m pytest tests/test_perrow_ad.py -v
then the full CPU suite, then ruff check on the touched files. Append a "Review
fixes" section to HANDOFF_ADRECON.md with the new pytest output and the measured
even-size relative error, as your LAST action write the file REVIEW_FIXES_DONE.md
at the worktree root containing a one-paragraph summary.
