"""
Entrena un detector binario por categoría (config.CATEGORIAS): racismo,
clasismo, odio a mujeres/LGBTI/política/discapacidad/apariencia/delincuencia,
e insulto_personal. Cada uno es un TF-IDF + Regresión Logística chico,
entrenado SOLO con las filas de la fuente que trae esa etiqueta -- nunca se
le asigna 0 a una fuente que no anotó esa categoría (ver el porqué en
config.py, sección DETECTORES DE CATEGORÍA).

Requiere: python preparar_datos.py (reutiliza su split train/test, para no
inventar una partición nueva que sí tenga fuga de datos entre ambos).

    python entrenar_categorias.py

Guarda todos los detectores juntos en salidas/detectores_categoria.joblib
(diccionario nombre -> pipeline) y un resumen de soporte + F1 en
salidas/categorias_resultados.csv. Los usa votar.py para el "comité de
categorías" del ensamble.
"""

import csv

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

import config


def _leer_tsv(ruta):
    return pd.read_csv(ruta, sep="\t", quoting=csv.QUOTE_NONE, escapechar="\\", keep_default_na=False)


def _cargar_fuente_cruda(fuente):
    """None si falta esa fuente (en vez de tronar todo el script): así
    entrenar_categoria() puede omitir solo esa categoría y seguir con las
    demás, igual que descargar_datos.py omite solo la fuente que falte."""
    ruta = config.RUTA_CRUDOS / f"{fuente}.csv"
    if not ruta.exists():
        return None
    return pd.read_csv(ruta, keep_default_na=False).drop_duplicates(subset="texto")


def _etiqueta_binaria(df, columna_extra, valores_positivos):
    return df[f"extra_{columna_extra}"].isin(valores_positivos).astype(int)


def _particion_categoria(nombre_split_tsv, fuente, crudo):
    """Filtra entrenamiento.tsv o prueba.tsv a solo esa fuente, y lo une con
    su tabla cruda (que trae la columna extra_*) por texto exacto -- así se
    reutiliza el split ya sin fuga de preparar_datos.py en vez de armar una
    partición aparte para cada categoría."""
    split = _leer_tsv(config.RUTA_PROCESADOS / nombre_split_tsv)
    split = split[split["fuente"] == fuente]
    return split.merge(crudo, on="texto", how="inner", suffixes=("", "_crudo"))


def entrenar_categoria(nombre, info):
    fuente = info["fuente"]
    columna_extra, valores_positivos = info["columna_extra"], info["valores_positivos"]
    crudo = _cargar_fuente_cruda(fuente)
    if crudo is None:
        print(f"  {nombre}: falta data/crudos/{fuente}.csv, se omite "
              f"(corre primero: python descargar_datos.py --fuente {fuente})")
        return None, None

    train = _particion_categoria("entrenamiento.tsv", fuente, crudo)
    test = _particion_categoria("prueba.tsv", fuente, crudo)
    if len(train) == 0:
        print(f"  {nombre}: sin filas de '{fuente}' en entrenamiento.tsv, se omite "
              f"(¿corriste descargar_datos.py --fuente {fuente}?)")
        return None, None

    y_train = _etiqueta_binaria(train, columna_extra, valores_positivos)
    y_test = _etiqueta_binaria(test, columna_extra, valores_positivos)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=config.BASELINE_MAX_FEATURES,
            ngram_range=config.BASELINE_NGRAMA,
            sublinear_tf=True,
        )),
        ("clasificador", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=config.SEMILLA,
        )),
    ])
    pipeline.fit(train["texto"], y_train)

    f1_prueba = f1_score(y_test, pipeline.predict(test["texto"])) if y_test.sum() > 0 else float("nan")
    resumen = {
        "categoria": nombre, "fuente": fuente,
        "positivos_train": int(y_train.sum()), "total_train": len(y_train),
        "positivos_test": int(y_test.sum()), "total_test": len(y_test),
        "f1_test": round(f1_prueba, 4),
    }
    print(f"  {nombre}: {y_train.sum():,}/{len(y_train):,} positivos en train, "
          f"F1 en test = {f1_prueba:.4f}")
    return pipeline, resumen


def main():
    print("Entrenando detectores de categoría (uno por categoría, solo con su fuente de origen)...")
    detectores = {}
    resumenes = []
    for nombre, info in config.CATEGORIAS.items():
        pipeline, resumen = entrenar_categoria(nombre, info)
        if pipeline is not None:
            detectores[nombre] = pipeline
            resumenes.append(resumen)

    if not detectores:
        raise SystemExit("No se entrenó ningún detector de categoría -- revisa que hayas descargado "
                          "las fuentes que usa config.CATEGORIAS (piuba_odio, offendes).")

    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    joblib.dump(detectores, config.RUTA_DETECTORES_CATEGORIA)
    pd.DataFrame(resumenes).to_csv(config.RUTA_SALIDAS / "categorias_resultados.csv", index=False)

    print(f"\nOK: {len(detectores)} detectores guardados en {config.RUTA_DETECTORES_CATEGORIA}")


if __name__ == "__main__":
    main()
