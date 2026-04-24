# Check multi-size correlation for REO
# Does having defects of ALL sizes simultaneously predict contamination?
reo = df[df.SUB_VENDOR == "REO"].copy()
reo["has_small"]   = (reo.count_small_excl_last   > 0).astype(int)
reo["has_xlarge"]  = (reo.count_xlarge_excl_last  > 0).astype(int)
reo["has_medium"]  = (reo.count_medium_excl_last  > 0).astype(int)
reo["multi_size"]  = (reo.has_small + reo.has_xlarge + reo.has_medium)

print(reo["multi_size"].corr(reo["contaminated"]))



print(df.groupby("SUB_VENDOR").apply(
    lambda x: x["defect_count_excl_last"].corr(x["contaminated"])
).round(3))


reo = df[df.SUB_VENDOR == "REO"].copy()
print(pd.crosstab(reo.predicted_defects,
                  reo.contaminated,
                  normalize="index").round(3) * 100)
