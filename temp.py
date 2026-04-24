# Check coverage distribution thresholds
for thresh in [0.05, 0.1, 0.2, 0.5, 1.0]:
    pct_below = (df["total_coverage_excl_last"] <= thresh).mean() * 100
    pct_above = 100 - pct_below
    n_above   = (df["total_coverage_excl_last"] > thresh).sum()
    print(f"threshold {thresh}%:  "
          f"{pct_below:.1f}% below  |  "
          f"{pct_above:.1f}% above  |  "
          f"n above = {n_above:,}")
