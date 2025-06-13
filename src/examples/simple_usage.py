import os
import sys
import pandas as pd

# permite importar pacotes do diret\u00f3rio pai
sys.path.append(os.path.abspath(".."))
from woe_guard import WOEGuard

# carrega dataset de exemplo
DATA_PATH = os.path.join("..", "data", "case_data_science_credit.csv")
df = pd.read_csv(DATA_PATH, sep=";")

categorical_cols = [
    "qtd_restritivos",
    "verificacao_fonte_de_renda",
    "qtd_atrasos_ultimos_2a",
    "qtd_consultas_ultimos_6m",
]

# instancia e aplica o WOEGuard
encoder = WOEGuard(categorical_cols=categorical_cols, drop_original=True)
encoded = encoder.fit_transform(df[categorical_cols], df["target"])
print(encoded.head())
