from scape.scape_t.ablations import submetric_ablation_dev, submetric_ablation_abstractness, layer_ablation_dev

b, df = submetric_ablation_dev(); print(f"\nDev baseline: {b:.2f}"); display(df)
b2, df2 = submetric_ablation_abstractness(); print(f"\nAbstractness baseline: {b2:.2f}"); display(df2)
layer_df, _ = layer_ablation_dev(); print("\nLayer ablations (Dev):"); display(layer_df)
