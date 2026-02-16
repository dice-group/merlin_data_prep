from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AllocationConfig:
    total_new_tokens: int = 128_000
    weight_ppw: float = 0.5
    weight_cpt: float = 0.5
    weight_unk: float = 0.0
    gamma: float = 2.0


def _safe_minmax(values: np.ndarray, *, invert: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.where(finite, arr, 0.0)
    min_val = float(arr.min())
    max_val = float(arr.max())
    denom = max_val - min_val
    if denom <= 0.0:
        scaled = np.zeros_like(arr, dtype=float)
    else:
        scaled = (arr - min_val) / denom
    if invert:
        scaled = 1.0 - scaled
    return scaled


def compute_allocation(metrics: pd.DataFrame, config: AllocationConfig) -> pd.DataFrame:
    df = metrics.copy()
    if df.empty:
        raise ValueError("Allocation metrics dataframe is empty.")

    df["ppw_n"] = _safe_minmax(df["pieces_per_word"].to_numpy(dtype=float))
    # lower chars-per-token is better
    df["cpt_n"] = _safe_minmax(df["chars_per_token"].to_numpy(dtype=float), invert=True)
    df["unk_n"] = _safe_minmax(df["unknown_rate"].to_numpy(dtype=float))

    # 1. linear combination
    df["I_lin"] = (
        config.weight_ppw * df["ppw_n"]
        + config.weight_cpt * df["cpt_n"]
        + config.weight_unk * df["unk_n"]
    )

    # 2. non-linear scaling to emphasize bad performing langs
    nonlinear = np.power(np.clip(df["I_lin"].to_numpy(dtype=float), a_min=0.0, a_max=None), config.gamma)
    df["inefficiency_score"] = _safe_minmax(nonlinear)

    # 3. allocate tokens proportionally to inefficiency scores (bad langs get more tokens)
    n_langs = len(df)
    scores = df["inefficiency_score"].to_numpy(dtype=float)
    finite_positive = np.isfinite(scores) & (scores > 0.0)
    if not finite_positive.any():
        base = config.total_new_tokens // n_langs
        alloc = np.full(n_langs, base, dtype=int)
        alloc[: config.total_new_tokens % n_langs] += 1
    else:
        score_sum = float(scores[finite_positive].sum())
        ideal = (scores / score_sum) * config.total_new_tokens
        alloc = np.floor(ideal).astype(int)

        shortfall = int(config.total_new_tokens - int(alloc.sum()))
        if shortfall > 0:
            remainders = ideal - alloc
            idx = np.argsort(-remainders)[:shortfall]
            alloc[idx] += 1
        elif shortfall < 0:
            remainders = ideal - alloc
            idx = np.argsort(remainders)[: abs(shortfall)]
            alloc[idx] -= 1

    df["token_alloc"] = alloc
    df = df.sort_values("inefficiency_score", ascending=False).reset_index(drop=True)
    return df[
        [
            "language",
            "pieces_per_word",
            "chars_per_token",
            "unknown_rate",
            "ppw_n",
            "cpt_n",
            "unk_n",
            "I_lin",
            "inefficiency_score",
            "token_alloc",
        ]
    ]
