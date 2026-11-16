"""
Junta las fuentes de data/crudos/, limpia, y arma train/val/test en
data/procesados/, más un set aparte de frases mixtas (code-switching
español/inglés) para medir -- nunca para entrenar -- qué tan bien generaliza
el modelo a modismos y mezcla de idiomas.

Requiere haber corrido antes: python descargar_datos.py

    python preparar_datos.py

Resultado (data/procesados/):
    entrenamiento.tsv      texto, idioma, ofensivo, fuente
    validacion.tsv         igual esquema
    prueba.tsv             igual esquema
    prueba_codeswitch.tsv  frases mixtas es+en, NUNCA vistas en entrenamiento
"""

import csv
import html
import random
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

import config


def normalizar(texto):
    # Davidson (inglés) trae tuits crudos con entidades HTML sin decodificar
    # (ej. "&amp;", "&#128514;" en vez de "&" y el emoji real) -- sin esto,
    # el modelo vería esas entidades como texto literal en vez del símbolo.
    texto = html.unescape(str(texto))
    texto = unicodedata.normalize("NFC", texto)
    return " ".join(texto.split())  # colapsa espacios/saltos de línea repetidos


def cargar_crudos():
    tablas = []
    for nombre in config.FUENTES:
        ruta = config.RUTA_CRUDOS / f"{nombre}.csv"
        if not ruta.exists():
            raise SystemExit(f"Falta {ruta}. Corre primero: python descargar_datos.py --fuente {nombre}")
        tablas.append(pd.read_csv(ruta, keep_default_na=False))
    return pd.concat(tablas, ignore_index=True)


def limpiar(df):
    df = df.copy()
    df["texto"] = df["texto"].map(normalizar)

    n_palabras = df["texto"].str.split().str.len()
    df = df[(n_palabras >= config.MIN_PALABRAS) & (n_palabras <= config.MAX_PALABRAS)]

    df = df[df["texto"].str.len() > 0]

    # Si el mismo texto exacto aparece con etiquetas distintas en fuentes
    # distintas (pasa poco, pero pasa), se descarta por ambiguo en vez de
    # quedarse con una de las dos al azar.
    conteos = df.groupby("texto")["ofensivo"].nunique()
    textos_ambiguos = set(conteos[conteos > 1].index)
    df = df[~df["texto"].isin(textos_ambiguos)]

    return df.drop_duplicates(subset="texto").reset_index(drop=True)


def particionar(df, semilla):
    """Train/val/test estratificado por (idioma, ofensivo), para que ningún
    idioma ni ninguna clase quede en desventaja en val/test."""
    estrato = df["idioma"] + "_" + df["ofensivo"].astype(str)
    train, resto = train_test_split(
        df, test_size=config.PROPORCION_VAL + config.PROPORCION_TEST,
        stratify=estrato, random_state=semilla,
    )
    estrato_resto = resto["idioma"] + "_" + resto["ofensivo"].astype(str)
    proporcion_test_en_resto = config.PROPORCION_TEST / (config.PROPORCION_VAL + config.PROPORCION_TEST)
    val, test = train_test_split(
        resto, test_size=proporcion_test_en_resto,
        stratify=estrato_resto, random_state=semilla,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def generar_codeswitch(df, n, semilla):
    """Concatena una frase en español con una en inglés (con un conector al
    azar) para simular code-switching. La frase resultante es ofensiva si
    CUALQUIERA de las dos mitades lo era -- igual que una persona real
    insertaría una grosería/modismo en un idioma dentro de una frase del
    otro. Se arma a partir de df (train O prueba, nunca mezclando ambos)."""
    aleatorio = random.Random(semilla)
    es = df[df["idioma"] == "es"].reset_index(drop=True)
    en = df[df["idioma"] == "en"].reset_index(drop=True)

    filas = []
    for _ in range(n):
        fila_es = es.iloc[aleatorio.randrange(len(es))]
        fila_en = en.iloc[aleatorio.randrange(len(en))]
        conector = aleatorio.choice(config.CONECTORES)
        if aleatorio.random() < 0.5:
            texto = f"{fila_es.texto}{conector}{fila_en.texto}"
        else:
            texto = f"{fila_en.texto}{conector}{fila_es.texto}"
        filas.append({
            "texto": texto,
            "idioma": "mixto",
            "ofensivo": max(fila_es.ofensivo, fila_en.ofensivo),
            "fuente": "sintetico_codeswitch",
        })
    return pd.DataFrame(filas)


def _guardar_tsv(df, ruta):
    df.to_csv(ruta, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")


def main():
    print("Cargando fuentes crudas...")
    df = cargar_crudos()
    print(f"  {len(df):,} filas crudas de {df['fuente'].nunique()} fuentes")

    print("Limpiando (normalización, longitud, duplicados, ambigüedades)...")
    df = limpiar(df)
    print(f"  {len(df):,} filas después de limpiar")
    print(df.groupby(["idioma", "ofensivo"]).size().to_string())

    print("Particionando en train/val/test (estratificado por idioma+etiqueta)...")
    train, val, test = particionar(df, config.SEMILLA)
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    print(f"Generando {config.N_CODESWITCH_ENTRENAMIENTO:,} frases mixtas de entrenamiento "
          f"(a partir de train)...")
    mixtas_train = generar_codeswitch(train, config.N_CODESWITCH_ENTRENAMIENTO, config.SEMILLA)
    train = pd.concat([train, mixtas_train], ignore_index=True).sample(frac=1, random_state=config.SEMILLA)

    print(f"Generando {config.N_CODESWITCH_PRUEBA:,} frases mixtas de prueba "
          f"(a partir de test, NUNCA entrenamiento)...")
    prueba_codeswitch = generar_codeswitch(test, config.N_CODESWITCH_PRUEBA, config.SEMILLA + 1)

    config.RUTA_PROCESADOS.mkdir(parents=True, exist_ok=True)
    _guardar_tsv(train, config.RUTA_PROCESADOS / "entrenamiento.tsv")
    _guardar_tsv(val, config.RUTA_PROCESADOS / "validacion.tsv")
    _guardar_tsv(test, config.RUTA_PROCESADOS / "prueba.tsv")
    _guardar_tsv(prueba_codeswitch, config.RUTA_PROCESADOS / "prueba_codeswitch.tsv")

    print(f"\nOK: conjuntos en {config.RUTA_PROCESADOS}")
    print(f"  entrenamiento.tsv      {len(train):,} filas "
          f"(incluye {len(mixtas_train):,} mixtas sintéticas)")
    print(f"  validacion.tsv         {len(val):,} filas")
    print(f"  prueba.tsv             {len(test):,} filas")
    print(f"  prueba_codeswitch.tsv  {len(prueba_codeswitch):,} filas (solo para evaluar)")


if __name__ == "__main__":
    main()
