import os
import sys
import pandas as pd

# permite importar modulos da pasta pai
sys.path.append(os.path.abspath(".."))
from pipelines.training import build_pipeline

DATA_PATH = os.path.join("..", "data", "case_data_science_credit.csv")
df = pd.read_csv(DATA_PATH, sep=";")

categorical_cols = [
    "qtd_restritivos",
    "verificacao_fonte_de_renda",
    "qtd_atrasos_ultimos_2a",
    "qtd_consultas_ultimos_6m",
]

y = df["target"]
X = df.drop(columns=["target"])

pipeline = build_pipeline(categorical_cols)
pipeline.fit(X, y)

print("Pipeline treinado com sucesso")
