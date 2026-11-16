"""
Entrena un baseline rápido (TF-IDF + Regresión Logística) sobre
entrenamiento.tsv, para tener una referencia barata en CPU antes de gastar
tiempo de GPU en el Transformer -- mismo espíritu que Clasificadores.py al
comparar varios modelos de sklearn.

Requiere: python preparar_datos.py

    python entrenar_baseline.py
"""

import csv
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config


def _leer_tsv(ruta):
    return pd.read_csv(ruta, sep="\t", quoting=csv.QUOTE_NONE, escapechar="\\", keep_default_na=False)


def main():
    print("Cargando entrenamiento.tsv...")
    train = _leer_tsv(config.RUTA_PROCESADOS / "entrenamiento.tsv")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=config.BASELINE_MAX_FEATURES,
            ngram_range=config.BASELINE_NGRAMA,
            sublinear_tf=True,
        )),
        ("clasificador", LogisticRegression(
            class_weight="balanced",  # las fuentes tienen más no-ofensivo que ofensivo
            max_iter=1000,
            random_state=config.SEMILLA,
        )),
    ])

    print(f"Entrenando TF-IDF (max_features={config.BASELINE_MAX_FEATURES:,}, "
          f"ngramas={config.BASELINE_NGRAMA}) + Regresión Logística sobre {len(train):,} ejemplos...")
    pipeline.fit(train["texto"], train["ofensivo"])

    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.RUTA_BASELINE)
    print(f"OK: baseline guardado en {config.RUTA_BASELINE}")


if __name__ == "__main__":
    main()
