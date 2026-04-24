# Logistic regression or simple correlation
# which size category best predicts contamination?
df["contaminated"] = (df["total_coverage_excl_last"] > 0.05).astype(int)

for sz in ["small","medium","large","xlarge","xxlarge"]:
    col  = f"count_{sz}_excl_last"
    corr = df[df.SUB_VENDOR=="REO"][col].corr(
           df[df.SUB_VENDOR=="REO"]["contaminated"])
    print(f"REO  {sz:8s} → contamination corr: {corr:.3f}")


df["week"] = pd.to_datetime(df["serial_dt"]).dt.to_period("W").dt.start_time

print(df.groupby(["week","SUB_VENDOR"])[[
    "count_small_excl_last",
    "count_xlarge_excl_last",
    "count_xxlarge_excl_last",
    "total_coverage_excl_last"
]].mean().round(2))


print(df.groupby(["SUB_VENDOR","predicted_defects"])
      ["count_small_excl_last"].mean().round(1).unstack())
