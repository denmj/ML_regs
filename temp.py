for thresh in [0.05, 0.1, 1.0]:
    print(f"\nThreshold: {thresh}%")
    print(df.groupby("SUB_VENDOR")["total_coverage_excl_last"]
          .apply(lambda x: (x > thresh).mean() * 100)
          .round(1))
