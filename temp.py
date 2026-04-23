size_ce_cols = ["clarck_evans_r_small", "clarck_evans_r_medium",
                "clarck_evans_r_large", "clarck_evans_r_xlarge",
                "clarck_evans_r_xxlarge"]
print(df.groupby("SUB_VENDOR")[size_ce_cols].median().round(3))

# 4C/4D — nn overall per vendor
print(df.groupby("SUB_VENDOR")[
    ["nn_mean_nnd", "nn_std_nnd"]
].agg(["mean","median"]).round(2))

# 4E — nn by size per vendor
size_nnd_cols = ["mean_nnd_small", "mean_nnd_medium", "mean_nnd_large",
                 "mean_nnd_xlarge", "mean_nnd_xxlarge"]
print(df.groupby("SUB_VENDOR")[size_nnd_cols].median().round(2))
