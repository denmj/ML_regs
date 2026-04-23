# Check serial_dt format
print(df["serial_dt"].dtype)
print(df["serial_dt"].head(3))

# Weekly class mix
df["week"] = pd.to_datetime(df["serial_dt"]).dt.to_period("W").dt.start_time
print(df.groupby(["week", "SUB_VENDOR", "predicted_defects"])
      .size()
      .unstack(fill_value=0)
      .head(6))
