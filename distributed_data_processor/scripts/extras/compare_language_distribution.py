#!/usr/bin/env python3
import pandas as pd

ORIG_PATH = "/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/lang_counts/language_sample_counts.tsv"
REAL_PATH = "/scratch/hpc-prf-merlin/joel/moe-study/data_prep/merlin_data_prep/distributed_data_processor/scripts/extras/lang_stats.tsv"
EXCLUDE = {"english"}

orig = pd.read_csv(ORIG_PATH, sep="\t", names=["lang", "samples"])
real = pd.read_csv(REAL_PATH, sep="\t")
real = real[real["lang"] != "TOTAL"]
if EXCLUDE:
    orig = orig[~orig["lang"].isin(EXCLUDE)]
    real = real[~real["lang"].isin(EXCLUDE)]

df = orig.merge(real, on="lang", how="inner", suffixes=("_orig", "_real"))
df["p_orig"] = df["samples_orig"] / df["samples_orig"].sum()
df["p_real"] = df["samples_real"] / df["samples_real"].sum()
df["diff"] = df["p_real"] - df["p_orig"]
df["sample_diff"] = df["samples_real"] - df["samples_orig"]

tv = 0.5 * df["diff"].abs().sum()
print(f"TV distance (samples): {tv:.6f}")
print(df.sort_values("diff", key=lambda s: s.abs(), ascending=False).head(30)[
    ["lang", "samples_orig", "samples_real", "sample_diff", "p_orig", "p_real", "diff"]
])
