"""
WoEEvaluator
============

Classe para avaliar a adequação da codificação WoE (Weight of Evidence) em
variáveis categóricas de um DataFrame para problemas de classificação binária.
Gera um relatório indicando se a variável apresenta monotonicidade adequada
entre WoE e taxa de default e sugere codificações alternativas quando WoE não é
recomendada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Sequence, Set


class EncodingManager:
    """
    Avalia se a aplicação de WoE é apropriada para variáveis categóricas.

    Parâmetros
    ----------
    id_cols : Sequence[str], opcional
        Colunas de identificadores que devem ser ignoradas.
    date_cols : Sequence[str], opcional
        Colunas de data que devem ser ignoradas.
    ignore_cols : Sequence[str], opcional
        Quaisquer outras colunas a serem ignoradas.
    min_pct : float, default=0.01
        Percentual mínimo de observações por categoria para não ser considerada esparsa.
    eps : float, default=1e-6
        Pequeno valor para evitar divisões por zero e logs indefinidos.
    """

    def __init__(
        self,
        id_cols: Optional[Sequence[str]] = None,
        date_cols: Optional[Sequence[str]] = None,
        ignore_cols: Optional[Sequence[str]] = None,
        min_pct: float = 0.01,
        eps: float = 1e-6,
    ) -> None:
        self.id_cols: Set[str] = set(id_cols or [])
        self.date_cols: Set[str] = set(date_cols or [])
        self.ignore_cols: Set[str] = set(ignore_cols or [])
        self.min_pct = float(min_pct)
        self.eps = float(eps)

        self._result: Optional[pd.DataFrame] = None

    # # --------------------------------------------------------------------- #
    # # Métricas internas
    # # --------------------------------------------------------------------- #
    # def _compute_woe(self, target: pd.Series, cat: pd.Series) -> pd.DataFrame:
    #     """Calcula WoE, taxa de evento e contagem por categoria."""
    #     df = pd.DataFrame({"target": target, "cat": cat}).copy()
    #     df["cat"] = df["cat"].fillna("MISSING")

    #     total_good = (df["target"] == 0).sum()
    #     total_bad = (df["target"] == 1).sum()

    #     grouped = (
    #         df.groupby("cat")["target"]
    #         .agg(["count", "sum"])
    #         .rename(columns={"sum": "bad"})
    #     )
    #     grouped["good"] = grouped["count"] - grouped["bad"]
    #     grouped["dist_good"] = grouped["good"] / (total_good + self.eps)
    #     grouped["dist_bad"] = grouped["bad"] / (total_bad + self.eps)
    #     grouped["woe"] = np.log(
    #         (grouped["dist_good"] + self.eps) / (grouped["dist_bad"] + self.eps)
    #     )
    #     grouped["target_rate"] = grouped["bad"] / (grouped["count"] + self.eps)
    #     return grouped.reset_index()[["cat", "woe", "target_rate", "count"]]

    @staticmethod
    def _is_monotonic(arr: np.ndarray, tol: float = 0.0) -> bool:
        """Verifica se um vetor é monotônico crescente ou decrescente."""
        diffs = np.diff(arr)
        return bool((diffs >= -tol).all() or (diffs <= tol).all())

    # --------------------------------------------------------------------- #
    # Avaliação de cada feature
    # --------------------------------------------------------------------- #
    def _evaluate_feature(
        self, target: pd.Series, cat: pd.Series
    ) -> tuple[bool, pd.DataFrame]:
        table = self._compute_woe(target, cat)
        ordered = table.sort_values("woe")
        monotonic = self._is_monotonic(ordered["target_rate"].values)
        return monotonic, table

    # --------------------------------------------------------------------- #
    # Interface pública
    # --------------------------------------------------------------------- #
    def fit(self, df: pd.DataFrame, target_col: str) -> "WoEEvaluator":
        """
        Executa a avaliação em todas as variáveis categóricas elegíveis.

        Retorna
        -------
        self : WoEEvaluator
            Para permitir encadeamento (method chaining).
        """
        if target_col not in df.columns:
            raise ValueError(f"'{target_col}' não encontrado no DataFrame.")

        # Seleciona colunas categóricas elegíveis
        eligible = [
            c
            for c in df.columns
            if (
                (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))
                and c
                not in self.id_cols.union(self.date_cols, self.ignore_cols, {target_col})
            )
        ]

        results = []
        n_rows = len(df)

        for col in eligible:
            monotonic, stats = self._evaluate_feature(df[target_col], df[col])
            min_prop = stats["count"].min() / n_rows
            sparse = min_prop < self.min_pct

            if monotonic and not sparse:
                woe_ok = True
                reason = ""
                rec_enc = "WoE"
            else:
                woe_ok = False
                reasons = []
                if not monotonic:
                    reasons.append("não monotônico")
                if sparse:
                    reasons.append("muito esparso")
                reason = "; ".join(reasons)
                # Sugestão de codificador
                rec_enc = "TargetEncoding" if not sparse else "LeaveOneOut"

            results.append(
                {
                    "feature": col,
                    "woe_ok": woe_ok,
                    "reason": reason,
                    "recommended_encoding": rec_enc,
                }
            )

        self._result = pd.DataFrame(results)
        return self

    def get_report(self) -> pd.DataFrame:
        """Retorna o DataFrame com o resultado da avaliação."""
        if self._result is None:
            raise RuntimeError("Execute 'fit' antes de chamar 'get_report'.")
        return self._result.copy()

    # --------------------------------------------------------------------- #
    # Conformidade scikit‑learn (opcional)
    # --------------------------------------------------------------------- #
    def fit_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Compatibilidade com pipelines: executa 'fit' e devolve o relatório."""
        return self.fit(df, target_col).get_report()


__all__ = ["WoEEvaluator"]
