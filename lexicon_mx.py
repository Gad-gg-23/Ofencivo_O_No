"""
Genera ejemplos de entrenamiento SINTÉTICOS a partir de
lexicon/jerga_mexicana.csv (armado a mano por el equipo) para tapar el
hueco de vocabulario mexicano que ninguna fuente real cubre: piuba_odio es
de Twitter argentino y OffendES de redes juveniles en español general,
ninguna de las dos trae modismos como "naco" (clasismo) o "prieto"
(racismo) -- ver config.py, sección LÉXICO DE MODISMOS MEXICANOS.

Esto NO es un detector de palabras aparte (una regla "si contiene X, es
ofensivo" inflaría falsos positivos en palabras ambiguas como "wey" o
"naco" -- ver columna depende_del_contexto del CSV). En vez de eso, arma
pares (texto, etiqueta) que entrenar_categorias.py mezcla con los datos
reales, para que el TF-IDF + Regresión Logística de cada categoría
aprenda estas palabras con sus propios pesos:

- depende_del_contexto == "no" (nunca ambigua, ej. "nigger", "mongol",
  "pinche indio"): solo ejemplos OFENSIVOS -- no hay uso benigno que enseñar.
- depende_del_contexto == "si" (ambigua, ej. "naco", "wey", "cabrón"):
  ejemplos en AMBOS sentidos (insultante y vocativo/amistoso), para que el
  modelo aprenda a distinguir por el resto de la frase, no por la sola
  presencia de la palabra.
- categoria_principal/secundaria == "ninguna_no_ofensivo" (ej. "chido",
  "no mames", "a toda madre"): se omiten del entrenamiento por completo,
  tal como se pidió -- quedan en el CSV solo como referencia.
"""

import random

import pandas as pd

import config

PLANTILLAS_OFENSIVAS = [
    "{variante}",
    "pinche {variante}",
]

# Varias plantillas distintas (no una sola) para que el modelo no aprenda
# "oye X, ¿cómo estás?" como SU PROPIA categoría en vez de aprender que la
# palabra en un marco vocativo/amistoso no es ofensiva -- con una sola
# plantilla repetida cientos de veces, ese molde se vuelve la señal más
# fuerte del dataset sintético y contamina categorías sin relación.
PLANTILLAS_NEUTRAS = [
    "oye {variante}, ¿cómo estás?",
    "qué onda {variante}, nos vemos al rato",
    "gracias {variante}, te debo una",
    "{variante}, ya llegamos",
]


def cargar_lexicon():
    df = pd.read_csv(config.RUTA_LEXICON, keep_default_na=False)
    es_no_ofensiva = (
        (df["categoria_principal"] == config.NINGUNA_NO_OFENSIVA_LEXICON)
        | (df["categoria_secundaria"] == config.NINGUNA_NO_OFENSIVA_LEXICON)
    )
    return df[~es_no_ofensiva]


def _filas_de_categoria(df, categoria_csv):
    return df[(df["categoria_principal"] == categoria_csv) | (df["categoria_secundaria"] == categoria_csv)]


def generar_ejemplos(categoria_nombre, df_lexicon=None):
    """(texto, etiqueta) sintéticos para la categoría de config.CATEGORIAS
    con nombre `categoria_nombre` (ej. "clasismo", "odio_lgbti"), o lista
    vacía si esa categoría no tiene ninguna entrada en el léxico (ej.
    odio_apariencia, odio_delincuencia -- el CSV no las cubre)."""
    categoria_csv = next(
        (csv_nombre for csv_nombre, interno in config.MAPA_CATEGORIAS_LEXICON.items()
         if interno == categoria_nombre),
        None,
    )
    if categoria_csv is None:
        return []

    df = df_lexicon if df_lexicon is not None else cargar_lexicon()
    filas = _filas_de_categoria(df, categoria_csv)

    ejemplos = []
    for i, (_, fila) in enumerate(filas.iterrows()):
        variante = fila["variante"]
        es_frase = " " in variante.strip()

        if es_frase:
            # La variante ya es una expresión completa e inequívoca (ej.
            # "no seas naco", "pinche indio") -- usarla tal cual, sin
            # envolverla en otra plantilla (eso daría "pinche no seas
            # naco", gramaticalmente roto y una señal ruidosa para el
            # modelo). Una frase así nunca es un saludo amistoso, así que
            # siempre cuenta como ofensiva sin importar si la PALABRA raíz
            # es ambigua.
            ejemplos.append((variante, 1))
        elif fila["depende_del_contexto"] == "no":
            for plantilla in PLANTILLAS_OFENSIVAS:
                ejemplos.append((plantilla.format(variante=variante), 1))
        else:
            ejemplos.append((PLANTILLAS_OFENSIVAS[1].format(variante=variante), 1))
            plantilla_neutra = PLANTILLAS_NEUTRAS[i % len(PLANTILLAS_NEUTRAS)]
            ejemplos.append((plantilla_neutra.format(variante=variante), 0))

    if len(ejemplos) > config.MAX_EJEMPLOS_LEXICON_POR_CATEGORIA:
        random.Random(config.SEMILLA).shuffle(ejemplos)
        ejemplos = ejemplos[:config.MAX_EJEMPLOS_LEXICON_POR_CATEGORIA]

    return ejemplos
