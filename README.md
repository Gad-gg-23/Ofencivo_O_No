# Ofensivo / No Ofensivo (español + inglés)

Práctica de Machine Learning (ESCOM, IPN). Entra una frase u oración y el
modelo dice si es **ofensiva** (groserías, insultos, racismo, clasismo, etc.)
o **no ofensiva**. Funciona en español, en inglés, y con frases que mezclan
ambos idiomas (code-switching/modismos).

## Inicio rápido

```bash
pip install -r requirements.txt
python Ofensivo_o_NoOfensivo.py descargar           # 4 fuentes desde Hugging Face (una es gated, ver abajo)
python Ofensivo_o_NoOfensivo.py preparar            # limpia, une y arma train/val/test + code-switching
python Ofensivo_o_NoOfensivo.py entrenar-base       # baseline TF-IDF + Regresión Logística (con K-Fold, CPU)
python Ofensivo_o_NoOfensivo.py entrenar-categorias # detectores de racismo/clasismo/etc. (CPU)
python Ofensivo_o_NoOfensivo.py entrenar            # afina XLM-RoBERTa (necesita GPU, ver colab/)
python Ofensivo_o_NoOfensivo.py evaluar             # compara baseline vs. Transformer vs. ensamble
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

`piuba_odio` y `OffendES` además guardan, en `data/crudos/` (no en los TSV de arriba), sus
columnas de categoría originales (`extra_RACISM`, `extra_CLASS`, ..., `extra_label`) —
las usa `entrenar_categorias.py` para los detectores de racismo/clasismo/etc. (ver más abajo).

## Estructura

```
Ofensivo_o_NoOfensivo.py   punto de entrada (descargar / preparar / entrenar-base / entrenar-categorias / entrenar / evaluar / votar / predecir)
config.py                  fuentes de datos, categorías, modelo e hiperparámetros, todo centralizado
descargar_datos.py         baja las 4 fuentes de Hugging Face y unifica su esquema (+ columnas de categoría)
preparar_datos.py          limpieza, partición y code-switching sintético
entrenar_baseline.py       TF-IDF + Regresión Logística, con K-Fold para elegir C (referencia rápida en CPU)
entrenar_categorias.py     detectores binarios por categoría (racismo, clasismo, insulto_personal, etc.)
entrenar.py                 afina XLM-RoBERTa con pérdida ponderada por clase
evaluar.py                  compara baseline vs. Transformer vs. ensamble (general, por idioma/fuente, code-switching, con IC bootstrap)
votar.py                    ensamble por votación (baseline + Transformer + comité de categorías)
predecir.py                  clasifica una frase (usa el ensamble si está completo); --explicar muestra qué palabras influyeron más
colab/Ofensivo_Colab.ipynb  notebook para correr todo con GPU gratuita
GUIA_EQUIPO.md              guía paso a paso para el equipo (Colab, Hugging Face, troubleshooting)
data/crudos/                una tabla unificada por fuente            (ignorado por Git)
data/procesados/            conjuntos listos para entrenar
salidas/                     modelos, baseline, detectores y métricas  (los modelos, ignorados por Git)
```

## Modelos

### Baseline: TF-IDF + Regresión Logística (`entrenar_baseline.py`)

Bolsa de palabras (unigrams + bigrams, hasta 50,000 features) + regresión logística con
`class_weight="balanced"`. Entrena en segundos en CPU. Sirve como referencia: si el
Transformer no le gana por un margen claro, no vale la pena la complejidad ni el tiempo de GPU
— misma lógica que `Clasificadores.py` al comparar varios modelos de sklearn.

**Validación (K-Fold):** antes de entrenar el final, se prueban 3 valores de `C`
(`config.VALORES_C_BASELINE`) con 5-Fold estratificado (`config.K_FOLDS_BASELINE`), igual que en
la práctica de `Validacion` del curso, y se reporta F1 promedio ± desviación entre folds en vez
de un solo número de un solo split (`salidas/baseline_cv_resultados.csv`). En nuestras pruebas,
`C=10` le ganó a `C=1` por ~1.5 puntos de F1 — la búsqueda sí valió la pena. **No se hace lo mismo
con el Transformer**: con ~125k ejemplos, reentrenar 5 veces en GPU para K-Fold completo gastaría
de más las horas gratis de Colab por una ganancia marginal frente a un split bien estratificado
(ver más abajo cómo sí se valida el Transformer sin reentrenar).

### Detectores de categoría (`entrenar_categorias.py`)

Un TF-IDF + Regresión Logística chico por cada categoría de `config.CATEGORIAS` (racismo,
clasismo, odio a mujeres/LGBTI/política/discapacidad/apariencia/delincuencia, e
insulto_personal), entrenado **solo con las filas de la fuente que trae esa etiqueta** —
`piuba_odio` para las 8 de odio hacia un grupo, `OffendES` para insultos a una persona
específica. No se le asigna "0" a una fuente que simplemente no anotó esa categoría (ej. no
sabemos si un tuit en inglés de `davidson` es clasista solo porque esa fuente no lo etiquetó;
asumir que no lo es metería ruido justo en la categoría que queremos detectar bien). Por esto
mismo, estos detectores **solo cubren español** por ahora — las fuentes con esas categorías
anotadas son ambas en español.

### Ensamble por votación (`votar.py`)

3 votantes independientes deciden si una frase es ofensiva:

1. **Baseline general** (arriba).
2. **Transformer general** (`entrenar.py`, ver abajo).
3. **Comité de categorías**: vota ofensivo si *cualquier* detector de categoría se activa —
   cuenta como un solo voto, no uno por categoría, para no sesgar la votación hacia "ofensivo"
   solo por tener varios detectores.

Ofensivo si al menos 2 de los 3 votan ofensivo (`config.VOTOS_MINIMOS_OFENSIVO`). La idea: el
baseline capta groserías/insultos explícitos aunque el Transformer dude por el contexto, el
Transformer capta ofensa implícita o sarcástica que el baseline (bolsa de palabras) no puede
ver, y el comité de categorías aporta una señal específica de *por qué* algo es ofensivo
(útil para explicar en la exposición, no solo para votar). `evaluar.py` reporta si el ensamble
le gana a cada modelo por separado; `predecir.py` lo usa automáticamente en cuanto las 3 piezas
estén entrenadas.

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

1. **General**: baseline vs. Transformer vs. ensamble sobre `prueba.tsv`.
2. **Validación (Bootstrap):** sobre el resultado general de cada modelo, 1,000 remuestreos con
   reemplazo de `prueba.tsv` (`config.N_BOOTSTRAP`) para dar un intervalo de confianza del 95% a
   Accuracy y F1, en vez de un solo número suelto — barato porque no reentrena nada, solo
   recalcula la métrica sobre cada remuestreo. Es la otra técnica de `Validacion` que sí es
   viable para el Transformer sin gastar horas de GPU.
3. **Por idioma y por fuente**: para detectar si cada modelo generaliza igual a los 4 datasets
   de origen o solo a algunos (ej. puede que le cueste más el dominio político de
   `piuba_odio` que los insultos directos de `OffendES`).
4. **Code-switching**: desempeño sobre `prueba_codeswitch.tsv` — la parte que justifica haber
   elegido un modelo multilingüe en vez de uno por idioma.

Resultados completos en `salidas/resultados_comparacion.csv`; matrices de confusión en
`salidas/matriz_confusion.csv` (Transformer) y `salidas/matriz_confusion_ensamble.csv`.

### Inferencia (`predecir.py` / `votar.py`)

```bash
python predecir.py --texto "eres un idiota"                                          # usa el ensamble si está completo
python predecir.py --texto "qué buen día, gracias por tu ayuda"
python predecir.py --texto "eres un idiota, you're such a loser" --explicar           # code-switching + interpretabilidad
python predecir.py --texto "eres un idiota" --sin-ensamble                           # fuerza solo el Transformer
python votar.py --texto "Q hdp! Estos chinos de mierda"                              # detalle de los 3 votos
```

`predecir.py` usa el ensamble de `votar.py` automáticamente en cuanto baseline, Transformer y
detectores de categoría están entrenados; si falta alguna pieza, cae de vuelta a usar solo el
Transformer y lo avisa. `--explicar` usa *leave-one-out* sobre el Transformer: quita cada
palabra, una a la vez, y mide cuánto cae la probabilidad de "ofensivo" al quitarla. Las
palabras con mayor caída son las que más influyeron en la predicción — una forma simple de
interpretabilidad sin librerías extra (ej. SHAP/LIME), suficiente para explicar en la
exposición por qué el modelo clasificó algo como ofensivo.
