# Guía para el equipo — Ofensivo / No Ofensivo (es + en)

Repositorio: **[github.com/Gad-gg-23/Ofencivo_O_No](https://github.com/Gad-gg-23/Ofencivo_O_No)**

Notebook de Colab (ábrelo directo con este link, no hace falta buscarlo):
**[colab.research.google.com/.../Ofensivo_Colab.ipynb](https://colab.research.google.com/github/Gad-gg-23/Ofencivo_O_No/blob/main/colab/Ofensivo_Colab.ipynb)**

Modelo afinado, compartido en Hugging Face (no en GitHub, pesa ~1.1 GB):
**[huggingface.co/GadiGG/ofensivo-no-ofensivo-escom](https://huggingface.co/GadiGG/ofensivo-no-ofensivo-escom)**

---

## Qué hace esta práctica, en una frase

Entra una frase en español, en inglés, o mezclando ambos (modismos/code-switching), y el
modelo dice si es **ofensiva** (groserías, insultos, racismo, clasismo, etc.) o **no
ofensiva**.

La decisión final no la da un solo modelo: un **ensamble por votación** combina el baseline, el
Transformer y un comité de detectores por categoría (racismo, clasismo, etc.) — ver
[Parte 7](#parte-7--ensamble-por-votación-votarpy).

**¿Solo quieres probar el modelo ya entrenado?** No hace falta Colab ni GPU — salta directo a
la [Parte 9](#parte-9--probar-con-tu-propia-oración). El notebook (Parte 1) solo es necesario
si vas a **re-entrenar o mejorar** el modelo.

---

## Parte 0 — Lo que hace el admin, una sola vez

1. **GitHub:** entra al repo → **Settings → Collaborators → Add people** → escribe el
   usuario/correo de GitHub de cada compañero → **Add**. Cada quien acepta la invitación antes
   de poder `git push`.
2. **Hugging Face (solo si alguien más va a subir el modelo entrenado, Parte 10):** entra a
   [huggingface.co/GadiGG/ofensivo-no-ofensivo-escom](https://huggingface.co/GadiGG/ofensivo-no-ofensivo-escom)
   → **Settings → Collaborators** → agrega su usuario de Hugging Face. Si el repo todavía no
   existe la primera vez que alguien corre la Parte 10, `push_to_hub` lo crea solo.
3. Comparte esta guía y el link del notebook de arriba con el equipo.

El resto de la guía es para **cada compañero**.

---

## Parte 1 — Correr el notebook (cada persona, todas las veces que entrene)

### 1.1 Abrir el notebook y activar GPU

1. Abre el link del notebook de arriba, inicia sesión con tu cuenta de Google si te lo pide.
2. `Entorno de ejecución` (menú de arriba) → `Cambiar tipo de entorno de ejecución` → **GPU**
   (la T4 gratuita) → **Guardar**.

### 1.2 Correr todo

`Entorno de ejecución → Ejecutar todas` (o celda por celda con ▶️, de arriba hacia abajo). Lo
que vas a ver pasar, en orden:

1. **Clona el repo** — rápido, segundos.
2. **Instala lo que falta y verifica la GPU** — `transformers`, `datasets`, `accelerate`,
   `huggingface_hub`. Debe imprimir `GPU disponible: True`. Si dice `False`, te faltó el paso
   1.1.2.
3. **Inicia sesión en Hugging Face** (Parte 3 del notebook) — necesario porque una de las 4
   fuentes (OffendES) es *gated*. Ver [Parte 2](#parte-2--sobre-offendes-gated) de esta guía.
4. **Descarga las 4 fuentes** (`Ofensivo_o_NoOfensivo.py descargar`) — un par de minutos.
5. **Prepara los datos** (`Ofensivo_o_NoOfensivo.py preparar`) — limpia, une las 4 fuentes,
   separa train/val/test (estratificado) y genera las frases sintéticas de code-switching.
   Unos minutos.
6. **Entrena el baseline** (`Ofensivo_o_NoOfensivo.py entrenar-base`) — TF-IDF + Regresión
   Logística, con 5-Fold para elegir el mejor `C` antes del entrenamiento final. Segundos en
   CPU.
7. **Entrena los detectores de categoría** (`Ofensivo_o_NoOfensivo.py entrenar-categorias`) —
   un modelo chico por racismo/clasismo/etc. Segundos en CPU.
8. **Afina el Transformer** (`Ofensivo_o_NoOfensivo.py entrenar`) — parte de
   `xlm-roberta-base` y lo afina con nuestros datos. Va a imprimir el progreso por época;
   **espera a que termine por completo** — al final debe decir `OK: modelo afinado en
   .../salidas/modelo_xlmr`. Tarda varios minutos en GPU.
9. **Evalúa** (`Ofensivo_o_NoOfensivo.py evaluar`) — esta es la prueba real de la práctica:
   Accuracy/Precision/Recall/F1 (con intervalo de confianza bootstrap) del baseline, el
   Transformer y el ensamble por votación sobre `prueba.tsv`, desglosado por idioma y por
   fuente, más el desempeño sobre las frases mixtas de `prueba_codeswitch.tsv`.
10. **Ver resultados** — imprime las tablas completas.
11. **Descargar o subir el modelo** (opcional) — ver Parte 10.
12. **Probar tu propia oración** (opcional) — ver Parte 9.

---

## Parte 2 — Sobre OffendES (gated)

A diferencia de las otras 3 fuentes (públicas, sin login), **OffendES pide aceptar
condiciones de uso antes de poder descargarla**, incluso por código:

1. Entra a [huggingface.co/datasets/SINAI/OffendES](https://huggingface.co/datasets/SINAI/OffendES)
   con tu cuenta de Hugging Face iniciada.
2. Acepta las condiciones de acceso (botón en la página). Se hace una sola vez por cuenta.
3. En el notebook (Parte 3) o en tu computadora (`huggingface-cli login` / `hf auth login`),
   inicia sesión con un token (basta permiso **Read**, en
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

Si se te olvida este paso, `descargar_datos.py` va a fallar específicamente en `offendes` con
un mensaje que te manda de vuelta a este paso — las otras 3 fuentes se descargan igual.

---

## Parte 3 — Cómo correrlo en tu computadora en vez de Colab (opcional)

Solo para los pasos de datos y el baseline (sin GPU, afinar el Transformer sería lentísimo):

```bash
cd Ofensivo_o_NoOfensivo
pip install -r requirements.txt
python Ofensivo_o_NoOfensivo.py descargar
python Ofensivo_o_NoOfensivo.py preparar
python Ofensivo_o_NoOfensivo.py entrenar-base
python Ofensivo_o_NoOfensivo.py entrenar-categorias
```

`entrenar.py` sí necesita GPU para terminar en un tiempo razonable — para eso está el
notebook.

---

## Parte 4 — De dónde salen los datos (por si alguien pregunta en la exposición)

- **4 fuentes de Hugging Face Datasets**, combinadas para cubrir español, inglés, odio
  político (racismo/clasismo explícitos) e insultos personales directos — ver la tabla
  completa en [README.md](README.md).
- Se colapsan las etiquetas originales de cada una a binario: ofensivo (1) / no ofensivo (0).
- Se limpian duplicados, frases demasiado cortas/largas, y casos donde el mismo texto aparece
  con etiquetas distintas en dos fuentes (se descartan por ambiguos).
- Partición 80/10/10 **estratificada por idioma + etiqueta**, para que ningún idioma ni
  ninguna clase quede en desventaja en validación/prueba.
- Se generan frases sintéticas de code-switching (español+inglés mezclados con un conector al
  azar) — 6,000 para entrenamiento, 800 aparte solo para evaluar, nunca mezcladas entre sí.

---

## Parte 5 — Por qué un Transformer multilingüe preentrenado y no uno desde cero (ni uno por idioma)

`xlm-roberta-base` ya "entiende" ~100 idiomas porque se entrenó con muchísimos más datos de
los que nosotros podríamos juntar en Colab, y comparte su vocabulario de subpalabras entre
todos esos idiomas. Eso tiene dos ventajas sobre entrenar desde cero o usar un modelo por
idioma:

1. **Generaliza a frases mezcladas** (code-switching/modismos) sin necesitar detectar el
   idioma de antemano ni mantener dos modelos separados — justo lo que pide la práctica.
2. **Afinarlo da mejores resultados con pocos datos y pocas horas de GPU** que entrenar un
   Transformer desde cero — es la misma idea de *transfer learning* que ya se usó con
   MobileNetV2 en `Frutas_Verduras`/`Comida` y con M2M100 en `Traductor`.

El baseline de TF-IDF + Regresión Logística (`entrenar_baseline.py`) sirve para comprobar esto
con números: si el Transformer no le gana por un margen claro en `evaluar.py`, no habría valido
la pena la complejidad ni el tiempo de GPU.

---

## Parte 6 — Validación: K-Fold y Bootstrap

Dos técnicas de la práctica de `Validacion` del curso, cada una donde sí es barata:

- **K-Fold (baseline, `entrenar_baseline.py`):** antes del entrenamiento final, se prueban 3
  valores de `C` con 5-Fold estratificado, y se elige el que dé mejor F1 promedio entre folds
  (`salidas/baseline_cv_resultados.csv` trae el detalle). Barato porque es CPU.
- **Bootstrap (todos los modelos, `evaluar.py`):** 1,000 remuestreos con reemplazo de
  `prueba.tsv` para dar un intervalo de confianza del 95% a Accuracy y F1, sin reentrenar nada
  — solo se recalcula la métrica sobre cada remuestreo.

**No se hace K-Fold completo sobre el Transformer**: con ~125k ejemplos, reentrenar 5 veces en
GPU gastaría de más las horas gratis de Colab por una ganancia marginal frente a un split bien
estratificado. El Bootstrap de `evaluar.py` sí aplica al Transformer (y al ensamble) porque no
reentrena nada.

---

## Parte 7 — Ensamble por votación (`votar.py`)

3 votantes deciden juntos si una frase es ofensiva, en vez de confiar en un solo modelo:

1. **Baseline general.**
2. **Transformer general.**
3. **Comité de categorías** (`entrenar_categorias.py`): vota ofensivo si *cualquier* detector
   de racismo/clasismo/odio a mujeres/LGBTI/política/discapacidad/apariencia/delincuencia/
   insulto_personal se activa — cuenta como un solo voto, no uno por categoría.

Ofensivo si al menos 2 de los 3 votan ofensivo. Cada detector de categoría se entrena **solo
con la fuente que trae esa etiqueta** (`piuba_odio` para las 8 de odio hacia un grupo,
`OffendES` para insulto_personal) — por eso, por ahora, el comité de categorías **solo cubre
español**; en inglés el ensamble sigue funcionando por los otros 2 votantes.

```bash
python votar.py --texto "Q hdp! Estos chinos de mierda"
```

imprime la etiqueta final, cuántos votos de 3 la respaldan, y el detalle de cada votante
(incluida qué categorías específicas se activaron) — útil para explicar en la exposición no
solo *que* algo se marcó ofensivo, sino *por qué* (ej. "racismo" + "Transformer").

`evaluar.py` reporta si el ensamble le gana a cada modelo por separado; `predecir.py` lo usa
automáticamente en cuanto las 3 piezas estén entrenadas (si falta alguna, cae de vuelta a usar
solo el Transformer y lo avisa).

---

## Parte 8 — Interpretabilidad: `--explicar`

`predecir.py --explicar` quita cada palabra de la frase, una a la vez, y mide cuánto cae la
probabilidad de "ofensivo" según el Transformer al quitarla (*leave-one-out*). Las palabras con
mayor caída son las que más "convencieron" al modelo — útil para explicar en la exposición por
qué el modelo clasificó algo como ofensivo, sin necesitar librerías de interpretabilidad como
SHAP o LIME.

---

## Parte 9 — Probar con tu propia oración

**No hace falta Colab, GPU, ni entrenar nada para esto** — el modelo del equipo ya está en
Hugging Face. `predecir.py` lo busca en este orden: 1) una copia local en
`salidas/modelo_xlmr/` (si tú entrenaste), 2) si no hay, lo descarga solo de
[huggingface.co/GadiGG/ofensivo-no-ofensivo-escom](https://huggingface.co/GadiGG/ofensivo-no-ofensivo-escom)
(la primera vez tarda por el tamaño, ~1.1 GB; luego queda en caché). Para usar el ensamble
completo (Parte 7) en vez de solo el Transformer, también necesitas haber corrido
`entrenar-base` y `entrenar-categorias` localmente (son rápidos, no necesitan GPU).

```bash
cd Ofensivo_o_NoOfensivo
pip install -r requirements.txt

python predecir.py --texto "eres un idiota"
python predecir.py --texto "qué buen día, gracias por tu ayuda"
python predecir.py --texto "eres un idiota, you're such a loser" --explicar
```

---

## Parte 10 — Compartir el modelo entrenado con el equipo (opcional)

El Transformer afinado pesa **~1.1 GB** — **no va a GitHub**, que rechaza archivos de más de
100 MB. Por eso se comparte en Hugging Face, igual que el modelo de `Traductor`.

**Opción A — descargarlo a tu computadora:** la celda de la Parte 10 del notebook comprime
`salidas/modelo_xlmr/` en un .zip y lo descarga.

**Opción B — subirlo a Hugging Face** (recomendado):
1. Ve a https://huggingface.co/settings/tokens
2. **Create new token** → tipo **Write** → nómbralo (ej. "Colab Ofensivo") → **Create token**.
3. Cópialo — solo se muestra una vez.
4. Pégalo en el campo de la celda (no se ve mientras escribes, es normal).

A diferencia de un repo de GitHub, **no hace falta que el admin lo cree primero**:
`push_to_hub` crea `GadiGG/ofensivo-no-ofensivo-escom` automáticamente la primera vez que
alguien lo corre con un token válido. Si tu cuenta no es colaboradora todavía, pídele al admin
que te agregue (Parte 0.2).

⚠️ El token es como una contraseña: no lo compartas ni lo pegues en chats. Si se expone,
revócalo en esa misma página y genera uno nuevo.

Para que cualquiera use el modelo ya subido (sin tener que entrenarlo de nuevo):

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

modelo = AutoModelForSequenceClassification.from_pretrained("GadiGG/ofensivo-no-ofensivo-escom")
tokenizer = AutoTokenizer.from_pretrained("GadiGG/ofensivo-no-ofensivo-escom")
```

---

## Problemas comunes

| Error | Causa | Solución |
|---|---|---|
| `GPU disponible: False` | No activaste GPU | `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU` |
| `Error descargando offendes (...): ...` con aviso de condiciones/login | No aceptaste las condiciones de OffendES o no iniciaste sesión | Ver [Parte 2](#parte-2--sobre-offendes-gated) |
| `No hay modelo afinado en .../modelo_xlmr` (al correr `evaluar.py` o `predecir.py`) | No corriste `entrenar.py` todavía, o se corrió en otra sesión de Colab que ya se cerró | Corre `python Ofensivo_o_NoOfensivo.py entrenar` primero |
| `[Errno 2] No such file or directory: '/content/repo'` | Colab reinició la sesión (VM nueva), se perdió todo lo descargado | `Entorno de ejecución → Ejecutar todas` desde el principio |
| El `push_to_hub` de la Parte 10 falla con error de permisos | Tu cuenta de Hugging Face no es colaboradora del repo, o el token no tiene permiso Write | Pide que te agreguen como colaborador (Parte 0.2), o genera un token nuevo con permiso **Write** |
| `F1 en test = nan` al correr `entrenar_categorias.py` | Esa categoría no tiene ningún ejemplo positivo en `prueba.tsv` (pasa con categorías raras, ej. `odio_discapacidad`) | No es un bug; revisa `salidas/categorias_resultados.csv` para ver cuántos positivos tiene esa categoría en train/test |
| `votar.py`/`predecir.py` dicen que faltan componentes para el ensamble | No corriste `entrenar-base` y/o `entrenar-categorias` (además de `entrenar`) | Corre los 3: `entrenar-base`, `entrenar-categorias`, `entrenar` |
| F1 de la clase "ofensivo" mucho más bajo que el de "no ofensivo" | Normal dado el desbalance de clases (hay más no-ofensivo en casi todas las fuentes); `entrenar.py` ya usa pesos por clase para mitigarlo, pero no lo elimina del todo | No es un bug; si quieres mejorar, sube `EPOCAS` en `config.py` o revisa `salidas/matriz_confusion.csv` para ver qué tipo de error domina |
