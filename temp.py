no_def = df[df.predicted_defects == "no defect"]
print(no_def.groupby("SUB_VENDOR")["total_coverage_excl_last"]
      .describe().round(4))
print(f"\n% no_defect mirrors with coverage > 0.05%:")
print(no_def.groupby("SUB_VENDOR")["total_coverage_excl_last"]
      .apply(lambda x: (x > 0.05).mean() * 100).round(1))
