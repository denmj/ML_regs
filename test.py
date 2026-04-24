# Small defect clark evans distribution
col = "clarck_evans_r_small"

print(df.groupby("SUB_VENDOR")[col].describe().round(3))

# Pattern breakdown per vendor
for vendor in ["REO", "GH"]:
    vdf = df[df.SUB_VENDOR == vendor][col].dropna()
    clustered  = (vdf < 0.7).mean()  * 100
    random     = ((vdf >= 0.7) & (vdf <= 1.3)).mean() * 100
    dispersed  = (vdf > 1.3).mean()  * 100
    print(f"\n{vendor}:")
    print(f"  Clustered  (CE-R < 0.7):  {clustered:.1f}%")
    print(f"  Random     (0.7-1.3):     {random:.1f}%")
    print(f"  Dispersed  (CE-R > 1.3):  {dispersed:.1f}%")
