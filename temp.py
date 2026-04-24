print(df.groupby(["SUB_VENDOR", "predicted_defects"]).agg(
    mean_coverage  = ("total_coverage_excl_last", "mean"),
    median_coverage= ("total_coverage_excl_last", "median"),
    mean_count     = ("defect_count_excl_last",   "mean"),
    median_count   = ("defect_count_excl_last",   "median"),
    pct_clean      = ("total_coverage_excl_last",
                      lambda x: (x < 0.05).mean() * 100),
    n              = ("total_coverage_excl_last",  "count")
).round(3))

# 5E
print(df.groupby(["SUB_VENDOR", "predicted_defects"])
      ["worst_ring"].median().round(2).unstack())
