"""
Configuración central del clasificador Ofensivo / No Ofensivo (español + inglés).

Un solo lugar con rutas, fuentes de datos, modelo e hiperparámetros, para que
descargar_datos.py, preparar_datos.py, entrenar.py/entrenar_baseline.py y
evaluar.py no se desincronicen.

La idea de la práctica: entra una frase u oración (en español, en inglés, o
mezclando ambos con modismos/code-switching) y el modelo dice si es ofensiva
(groserías, insultos, racismo, clasismo, etc.) o no. Se combina un dataset de
odio político argentino (que ya etiqueta racismo/clasismo explícitamente), un
corpus de insultos juveniles en español, y dos corpus de lenguaje ofensivo en
inglés, para que el modelo generalice a ambos idiomas y no solo a un dominio.
"""

from pathlib import Path

# =========================================================
# RUTAS
# =========================================================

RAIZ = Path(__file__).resolve().parent

RUTA_DATOS = RAIZ / "data"
RUTA_CRUDOS = RUTA_DATOS / "crudos"          # una tabla unificada por fuente, tal cual se bajó de HF
RUTA_PROCESADOS = RUTA_DATOS / "procesados"  # train/val/test ya limpios y unidos

RUTA_SALIDAS = RAIZ / "salidas"

# =========================================================
# FUENTES DE DATOS (Hugging Face Datasets)
# =========================================================
# Cada fuente trae su propio esquema de columnas y etiquetas; aquí se dice
# cómo leerlas para que descargar_datos.py las deje todas en el mismo
# formato: texto, idioma, ofensivo (0/1), fuente.
#
# - Si la columna de la etiqueta es de tipo ClassLabel, se convierte el
#   entero a su nombre (ds.features[columna].int2str(valor)) ANTES de buscar
#   en mapa_label -- así el mapeo no depende de qué entero le tocó a cada
#   clase en esa versión del dataset, solo del nombre.
# - Si no es ClassLabel (texto plano, ej. OffendES), se usa el valor tal cual.
FUENTES = {
    "piuba_odio": {
        # Comentarios de Twitter en respuesta a noticias argentinas (Clarín,
        # Infobae, La Nación, Perfil, Crónica) durante la pandemia. Es la
        # ÚNICA fuente que ya etiqueta explícitamente RACISM y CLASS
        # (clasismo) además de HATEFUL, justo lo que pide la práctica.
        "repo": "piuba-bigdata/contextualized_hate_speech",
        "config": None,
        "idioma": "es",
        "columna_texto": "text",
        "columna_label": "HATEFUL",
        "mapa_label": {"Not Hateful": 0, "Hateful": 1},
        "gated": False,
        "splits": ["train", "dev", "test"],
        "cita": "Pérez et al., 2023, IEEE Access (arXiv:2210.00465)",
    },
    "offendes": {
        # Comentarios de jóvenes influencers en Twitter/Instagram/YouTube
        # (MeOffendES, IberLEF 2021). Más centrado en insultos y groserías
        # personales que en odio político -- complementa a piuba_odio.
        # GATED: hay que aceptar las condiciones en la página de HF antes de
        # poder descargarlo (ver GUIA_EQUIPO.md).
        "repo": "SINAI/OffendES",
        "config": None,
        "idioma": "es",
        "columna_texto": "comment",
        "columna_label": "label",
        # NO = ni ofensivo ni grosero, NOE = no ofensivo pero con groserías,
        # OFP = ofensivo contra una persona, OFG = ofensivo contra un grupo.
        "mapa_label": {"NO": 0, "NOE": 0, "OFP": 1, "OFG": 1},
        "gated": True,
        "splits": ["train", "validation", "test"],
        "cita": "Plaza-del-Arco et al., 2021, RANLP",
    },
    "tweet_eval_offensive": {
        # OffensEval 2019 (SemEval Task 6): tuits en inglés, ya binario.
        "repo": "cardiffnlp/tweet_eval",
        "config": "offensive",
        "idioma": "en",
        "columna_texto": "text",
        "columna_label": "label",
        "mapa_label": {"non-offensive": 0, "offensive": 1},
        "gated": False,
        "splits": ["train", "validation", "test"],
        "cita": "Zampieri et al., 2019, SemEval",
    },
    "davidson": {
        # Tuits en inglés con 3 clases; se colapsan a binario: tanto "hate
        # speech" como "offensive language" cuentan como ofensivo.
        "repo": "tdavidson/hate_speech_offensive",
        "config": None,
        "idioma": "en",
        "columna_texto": "tweet",
        "columna_label": "class",
        "mapa_label": {"hate speech": 1, "offensive language": 1, "neither": 0},
        "gated": False,
        "splits": ["train"],
        "cita": "Davidson et al., 2017, ICWSM",
    },
}

# =========================================================
# LIMPIEZA Y PARTICIÓN (preparar_datos.py)
# =========================================================

SEMILLA = 42

MIN_PALABRAS = 2     # frases de 1 palabra casi no dan contexto para decidir
MAX_PALABRAS = 100    # tuits/comentarios largos atípicos, probablemente ruido

PROPORCION_VAL = 0.10
PROPORCION_TEST = 0.10  # el resto (0.80) es entrenamiento

# -- Code-switching sintético (frases que mezclan español e inglés) --
# Ningún dataset público trae oraciones mixtas etiquetadas, así que se
# generan concatenando una frase en español con una en inglés (ya
# etiquetadas por separado) con un conector al azar. La frase resultante es
# ofensiva si CUALQUIERA de las dos mitades lo era.
CONECTORES = [" y ", " pero ", ", and ", ", but ", " porque ", " aunque ", ", so ", " aka "]

N_CODESWITCH_ENTRENAMIENTO = 6000  # se agregan a train, para que el modelo aprenda el patrón
N_CODESWITCH_PRUEBA = 800          # set aparte, SOLO para medir (nunca se entrena con esto)

# =========================================================
# MODELO PRINCIPAL: Transformer multilingüe preentrenado
# =========================================================
# XLM-RoBERTa comparte vocabulario de subpalabras entre ~100 idiomas, por
# eso generaliza razonablemente bien a frases que mezclan español e inglés
# (code-switching) sin necesitar un modelo por idioma. Misma idea de
# transfer learning que ya se usó con MobileNetV2 (Frutas_Verduras/Comida) y
# M2M100 (Traductor): no se entrena desde cero, se afina.
MODELO_BASE = "xlm-roberta-base"
MAX_TOKENS = 128

ETIQUETAS = {0: "no_ofensivo", 1: "ofensivo"}

RUTA_MODELO_DIR = RUTA_SALIDAS / "modelo_xlmr"  # NO va a Git (~1.1 GB)
RUTA_HISTORICO_CSV = RUTA_SALIDAS / "historico_entrenamiento.csv"

# Repo de Hugging Face donde se comparte el modelo ya afinado (pesa más que
# el límite de 100 MB de GitHub). Mismo patrón que Traductor.
HF_REPO_MODELO = "GadiGG/ofensivo-no-ofensivo-escom"

EPOCAS = 3
TAMANO_LOTE = 16
ACUMULACION_GRADIENTE = 2  # lote efectivo = 32
TASA_APRENDIZAJE = 2e-5
PASOS_CALENTAMIENTO = 500

# =========================================================
# BASELINE: TF-IDF + Regresión Logística (entrenar_baseline.py)
# =========================================================
# Referencia rápida en CPU para justificar que el Transformer vale la
# inversión de tiempo de GPU -- mismo espíritu que Clasificadores.py al
# comparar varios modelos de sklearn.
RUTA_BASELINE = RUTA_SALIDAS / "baseline_tfidf_logreg.joblib"
BASELINE_MAX_FEATURES = 50_000
BASELINE_NGRAMA = (1, 2)

# =========================================================
# EVALUACIÓN (evaluar.py)
# =========================================================
RUTA_RESULTADOS_CSV = RUTA_SALIDAS / "resultados_comparacion.csv"
RUTA_MATRIZ_CONFUSION_CSV = RUTA_SALIDAS / "matriz_confusion.csv"
