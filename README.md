# Ofensivo / No Ofensivo (español + inglés)

Práctica de Machine Learning (ESCOM, IPN). Entra una frase u oración y el
modelo dice si es **ofensiva** (groserías, insultos, racismo, clasismo, etc.)
o **no ofensiva**. Funciona en español, en inglés, y con frases que mezclan
ambos idiomas (code-switching/modismos).

## Inicio rápido

```bash
pip install -r requirements.txt
python Ofensivo_o_NoOfensivo.py descargar       # 4 fuentes desde Hugging Face (una es gated, ver abajo)
python Ofensivo_o_NoOfensivo.py preparar        # limpia, une y arma train/val/test + code-switching
python Ofensivo_o_NoOfensivo.py entrenar-base   # baseline TF-IDF + Regresión Logística (CPU, segundos)
python Ofensivo_o_NoOfensivo.py entrenar        # afina XLM-RoBERTa (necesita GPU, ver colab/)
python Ofensivo_o_NoOfensivo.py evaluar         # compara ambos modelos, incluye code-switching
python Ofensivo_o_NoOfensivo.py predecir --texto "eres un idiota" --explicar
```

**El equipo entrena en Colab, no localmente** (no hay GPU en esta máquina): ver
[`colab/Ofensivo_Colab.ipynb`](colab/Ofensivo_Colab.ipynb) y la guía paso a paso en
[GUIA_EQUIPO.md](GUIA_EQUIPO.md).

## Datos

Se combinan 4 fuentes públicas de Hugging Face Datasets para que el modelo generalice a
ambos idiomas y a distintos estilos de ofensa (insultos personales, odio político,
racismo, clasismo):

| Fuente | Idioma | Qué es | Filas usadas | Acceso |
|---|---|---|---|---|
| [`piuba-bigdata/contextualized_hate_speech`](https://huggingface.co/datasets/piuba-bigdata/contextualized_hate_speech) | es | Respuestas a noticias argentinas en Twitter; la única fuente que ya etiqueta **RACISM** y **CLASS** (clasismo) explícitamente | ~57k | Público |
| [`SINAI/OffendES`](https://huggingface.co/datasets/SINAI/OffendES) | es | Comentarios de jóvenes influencers (Twitter/Instagram/YouTube); insultos y groserías personales | ~30k | **Gated** (aceptar condiciones en HF) |
| [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (config `offensive`) | en | Tuits, OffensEval 2019 (SemEval) | ~14k | Público |
| [`tdavidson/hate_speech_offensive`](https://huggingface.co/datasets/tdavidson/hate_speech_offensive) | en | Tuits con discurso de odio o lenguaje ofensivo | ~25k | Público |

Las etiquetas originales de cada fuente se colapsan a binario (`ofensivo`: 0/1) en
`descargar_datos.py` (ver `config.FUENTES` para el mapeo exacto de cada una).

### Limpieza y partición (`preparar_datos.py`)

- Normalización Unicode (NFC) y de espacios.
- Se descartan frases de menos de 2 palabras o más de 100, y duplicados.
- Si el mismo texto aparece con etiquetas distintas en dos fuentes, se descarta por ambiguo.
- Partición 80/10/10 (train/val/test), **estratificada por idioma + etiqueta** para que
  ningún idioma ni ninguna clase quede en desventaja en validación/prueba.

### Code-switching sintético

Ningún dataset público trae oraciones que mezclen español e inglés etiquetadas, así que se
generan concatenando una frase en español con una en inglés (ya etiquetadas por separado) con
un conector al azar (" y ", ", and ", " pero ", etc.). La frase resultante es ofensiva si
cualquiera de las dos mitades lo era.

- **6,000 frases mixtas se agregan a `entrenamiento.tsv`**, para que el modelo aprenda el patrón.
- **800 frases mixtas aparte en `prueba_codeswitch.tsv`**, construidas solo a partir de
  `prueba.tsv` — nunca entrenamiento — para medir honestamente qué tan bien generaliza el
  modelo a mezclas que nunca vio como mezcla real.

### Conjuntos resultantes (`data/procesados/`)

| Archivo | Contenido | ¿En Git? |
|---|---|---|
| `entrenamiento.tsv` | ~80% de los datos + 6,000 frases mixtas sintéticas | No (pesa demasiado) |
| `validacion.tsv` | ~10% de los datos | Sí |
| `prueba.tsv` | ~10% de los datos | Sí |
| `prueba_codeswitch.tsv` | 800 frases mixtas es+en (solo para evaluar) | Sí |

Columnas en todos: `texto`, `idioma` (`es`/`en`/`mixto`), `ofensivo` (0/1), `fuente`.

## Estructura

```
Ofensivo_o_NoOfensivo.py   punto de entrada (descargar / preparar / entrenar-base / entrenar / evaluar / predecir)
config.py                  fuentes de datos, modelo e hiperparámetros, todo centralizado
descargar_datos.py         baja las 4 fuentes de Hugging Face y unifica su esquema
preparar_datos.py          limpieza, partición y code-switching sintético
entrenar_baseline.py       TF-IDF + Regresión Logística (referencia rápida en CPU)
entrenar.py                 afina XLM-RoBERTa con pérdida ponderada por clase
evaluar.py                  compara baseline vs. Transformer (general, por idioma, por fuente, code-switching)
predecir.py                  clasifica una frase; --explicar muestra qué palabras influyeron más
colab/Ofensivo_Colab.ipynb  notebook para correr todo con GPU gratuita
GUIA_EQUIPO.md              guía paso a paso para el equipo (Colab, Hugging Face, troubleshooting)
data/crudos/                una tabla unificada por fuente            (ignorado por Git)
data/procesados/            conjuntos listos para entrenar
salidas/                     modelos, baseline y métricas              (los modelos, ignorados por Git)
```

## Modelos

### Baseline: TF-IDF + Regresión Logística (`entrenar_baseline.py`)

Bolsa de palabras (unigrams + bigrams, hasta 50,000 features) + regresión logística con
`class_weight="balanced"`. Entrena en segundos en CPU. Sirve como referencia: si el
Transformer no le gana por un margen claro, no vale la pena la complejidad ni el tiempo de GPU
— misma lógica que `Clasificadores.py` al comparar varios modelos de sklearn.

### Principal: XLM-RoBERTa afinado (`entrenar.py`)

**`xlm-roberta-base`** (Transformer preentrenado en ~100 idiomas con un vocabulario de
subpalabras compartido entre todos ellos), con una cabeza de clasificación binaria. Se eligió
un modelo multilingüe en vez de uno por idioma porque:

1. Un solo modelo cubre español, inglés **y frases que mezclan ambos** (code-switching) — ver
   la sección de arriba — sin necesitar detectar el idioma de antemano ni mantener dos modelos.
2. Es la misma idea de *transfer learning* que ya se usó con MobileNetV2
   (`Frutas_Verduras`/`Comida`) y M2M100 (`Traductor`): ya "sabe" idioma antes de empezar, así
   que afinarlo con nuestros ~125k ejemplos da mejores resultados que entrenar un Transformer
   desde cero con los mismos datos.

Las clases están desbalanceadas (más "no ofensivo" que "ofensivo" en casi todas las fuentes),
así que el entrenamiento usa una `CrossEntropyLoss` con pesos inversos a la frecuencia de cada
clase (`entrenar.py::calcular_pesos_clase`), en vez de dejar que el modelo gane accuracy solo
por predecir siempre la clase mayoritaria.

El modelo afinado se guarda en `salidas/modelo_xlmr/` — **no va a GitHub** (pesa ~1.1 GB, el
límite de GitHub es 100 MB por archivo). Se comparte en Hugging Face, mismo patrón que
`Traductor`: ver `config.HF_REPO_MODELO` y la Parte 8 de [GUIA_EQUIPO.md](GUIA_EQUIPO.md).

### Evaluación (`evaluar.py`)

Sobre `prueba.tsv` y `prueba_codeswitch.tsv`, nunca vistos en entrenamiento, con **Accuracy,
Precision, Recall y F1** (estándar de clasificación binaria, igual que en prácticas anteriores):

1. **General**: ambos modelos sobre `prueba.tsv`.
2. **Por idioma y por fuente**: para detectar si el modelo generaliza igual a los 4 datasets
   de origen o solo a algunos (ej. puede que le cueste más el dominio político de
   `piuba_odio` que los insultos directos de `OffendES`).
3. **Code-switching**: desempeño sobre `prueba_codeswitch.tsv` — la parte que justifica haber
   elegido un modelo multilingüe en vez de uno por idioma.

Resultados completos en `salidas/resultados_comparacion.csv`; matriz de confusión del
Transformer en `salidas/matriz_confusion.csv`.

### Inferencia (`predecir.py`)

```bash
python predecir.py --texto "eres un idiota"
python predecir.py --texto "qué buen día, gracias por tu ayuda"
python predecir.py --texto "eres un idiota, you're such a loser" --explicar   # code-switching + interpretabilidad
```

`--explicar` usa *leave-one-out*: quita cada palabra, una a la vez, y mide cuánto cae la
probabilidad de "ofensivo" al quitarla. Las palabras con mayor caída son las que más
influyeron en la predicción — una forma simple de interpretabilidad sin librerías extra
(ej. SHAP/LIME), suficiente para explicar en la exposición por qué el modelo clasificó algo
como ofensivo.
