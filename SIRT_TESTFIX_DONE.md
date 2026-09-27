# SIRT Test Fix Done

The fast-perp discriminating test now uses a realistic ±7.5° per-column sweep and verifies that the exact reconstruction strictly beats the same-data mid-angle reconstruction. The measured correlations were 0.906461596 exact and 0.814047158 mid-angle, giving a measured margin of 0.092414438; the enforced conservative margin is 0.05. The focused tests, full CPU suite, and Ruff check all passed.
