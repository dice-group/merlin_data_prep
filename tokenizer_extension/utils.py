from __future__ import annotations

import pandas as pd


def compare_metrics(base_df: pd.DataFrame, extended_df: pd.DataFrame) -> pd.DataFrame:
    """Compares metrics of base tokenizer and extended tokenizer."""
    base = base_df.rename(
        columns={
            "pieces_per_word": "ppw_base",
            "chars_per_token": "cpt_base",
            "unknown_rate": "unk_base",
        }
    )
    extended = extended_df.rename(
        columns={
            "pieces_per_word": "ppw_new",
            "chars_per_token": "cpt_new",
            "unknown_rate": "unk_new",
        }
    )
    df = pd.merge(base, extended, on="language", how="inner")
    df["ppw_diff"] = df["ppw_base"] - df["ppw_new"]
    df["cpt_diff"] = df["cpt_new"] - df["cpt_base"]
    df["unk_diff"] = df["unk_base"] - df["unk_new"]
    return df
