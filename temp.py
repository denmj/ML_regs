# How many mirrors per class per vendor
# Need enough samples per class to make radial curves meaningful
print(df.groupby(["SUB_VENDOR", "predicted_defects"])
      .size()
      .unstack(fill_value=0))

# Coverage ring columns confirm
ring_cols = [f"coverage_ring_{i}" for i in range(9)]
print(df.groupby("predicted_defects")[ring_cols]
      .median()
      .round(4))
