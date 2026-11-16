"""
Entrena un baseline rápido (TF-IDF + Regresión Logística) sobre
entrenamiento.tsv, para tener una referencia barata en CPU antes de gastar
tiempo de GPU en el Transformer -- mismo espíritu que Clasificadores.py al
comparar varios modelos de sklearn.

Antes de entrenar el final, hace K-Fold (config.K_FOLDS_BASELINE) sobre el
valor de C de la Regresión Logística -- misma técnica de la práctica de
Validacion del curso, aquí barata porque es CPU. Esto no se hace sobre el
Transformer (ver config.py, sección VALIDACIÓN).

Requiere: python preparar_datos.py

    python entrenar_baseline.py
"""

import csv

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import config


def _leer_tsv(ruta):
    return pd.read_csv(ruta, sep="\t", quoting=csv.QUOTE_NONE, escapechar="\\", keep_default_na=False)


def _construir_pipeline(C):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=config.BASELINE_MAX_FEATURES,
            ngram_range=config.BASELINE_NGRAMA,
            sublinear_tf=True,
        )),
        ("clasificador", LogisticRegression(
            C=C,
            class_weight="balanced",  # las fuentes tienen más no-ofensivo que ofensivo
            max_iter=1000,
            random_state=config.SEMILLA,
        )),
    ])


def elegir_mejor_C(textos, etiquetas):
    """K-Fold estratificado sobre cada valor de C en config.VALORES_C_BASELINE,
    usando F1 (no accuracy, por el desbalance de clases). Devuelve el C con
    mejor F1 promedio y una tabla con el detalle de los folds, para reportar
    la variación entre folds en vez de un solo número suelto."""
    kfold = StratifiedKFold(n_splits=config.K_FOLDS_BASELINE, shuffle=True, random_state=config.SEMILLA)

    filas = []
    for C in config.VALORES_C_BASELINE:
        puntajes = cross_val_score(_construir_pipeline(C), textos, etiquetas, cv=kfold, scoring="f1")
        filas.append({"C": C, "f1_promedio": puntajes.mean(), "f1_std": puntajes.std(),
                      **{f"f1_fold_{i + 1}": p for i, p in enumerate(puntajes)}})
        print(f"  C={C}: F1 = {puntajes.mean():.4f} +/- {puntajes.std():.4f} (folds: {puntajes.round(4).tolist()})")

    tabla = pd.DataFrame(filas)
    mejor_C = tabla.loc[tabla["f1_promedio"].idxmax(), "C"]
    return mejor_C, tabla


def main():
    print("Cargando entrenamiento.tsv...")
    train = _leer_tsv(config.RUTA_PROCESADOS / "entrenamiento.tsv")

    print(f"Eligiendo C con {config.K_FOLDS_BASELINE}-Fold estratificado "
          f"sobre {config.VALORES_C_BASELINE}...")
    mejor_C, tabla_cv = elegir_mejor_C(train["texto"], train["ofensivo"])
    print(f"  Mejor C: {mejor_C}")

    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    tabla_cv.to_csv(config.RUTA_BASELINE_CV_CSV, index=False)

    print(f"Entrenando TF-IDF (max_features={config.BASELINE_MAX_FEATURES:,}, "
          f"ngramas={config.BASELINE_NGRAMA}) + Regresión Logística (C={mejor_C}) "
          f"sobre {len(train):,} ejemplos...")
    pipeline = _construir_pipeline(mejor_C)
    pipeline.fit(train["texto"], train["ofensivo"])

    joblib.dump(pipeline, config.RUTA_BASELINE)
    print(f"OK: baseline guardado en {config.RUTA_BASELINE}")
    print(f"    detalle de la validación cruzada en {config.RUTA_BASELINE_CV_CSV}")


if __name__ == "__main__":
    main()
