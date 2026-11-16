"""
Evalúa el baseline, el Transformer, y el ensamble por votación de ambos +
el comité de categorías (votar.py) sobre prueba.tsv y prueba_codeswitch.tsv,
ninguno de los dos visto en entrenamiento.

    python evaluar.py

Guardado en salidas/:

1. Comparación baseline vs. Transformer vs. ensamble sobre prueba.tsv, con
   Accuracy, Precision, Recall y F1 (estándar de clasificación binaria) --
   para ver si votar entre los tres de verdad mejora sobre cualquiera solo.
2. Intervalo de confianza (bootstrap, config.N_BOOTSTRAP remuestreos) sobre
   Accuracy y F1 de cada modelo en el resultado general -- en vez de un solo
   número suelto, misma técnica de la práctica de Validacion del curso,
   aquí barata porque no reentrena nada.
3. Desglose por idioma (es/en) y por fuente, para ver si cada modelo
   generaliza igual a los 4 datasets de origen o solo a algunos.
4. Desempeño sobre prueba_codeswitch.tsv (frases mixtas es+en).
"""

import csv

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


def _leer_tsv(ruta):
    return pd.read_csv(ruta, sep="\t", quoting=csv.QUOTE_NONE, escapechar="\\", keep_default_na=False)


def _metricas(etiquetas, predichas):
    precision, recall, f1, _ = precision_recall_fscore_support(
        etiquetas, predichas, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": accuracy_score(etiquetas, predichas),
        "precision_ofensivo": precision,
        "recall_ofensivo": recall,
        "f1_ofensivo": f1,
    }


def intervalo_confianza_bootstrap(etiquetas, predichas):
    """Remuestrea (etiquetas, predichas) con reemplazo config.N_BOOTSTRAP
    veces y devuelve el intervalo de confianza de Accuracy y F1 -- no
    reentrena nada, solo recalcula la métrica sobre cada remuestreo."""
    etiquetas, predichas = np.asarray(etiquetas), np.asarray(predichas)
    rng = np.random.default_rng(config.SEMILLA)
    n = len(etiquetas)
    accs, f1s = np.empty(config.N_BOOTSTRAP), np.empty(config.N_BOOTSTRAP)
    for i in range(config.N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        accs[i] = accuracy_score(etiquetas[idx], predichas[idx])
        f1s[i] = f1_score(etiquetas[idx], predichas[idx], pos_label=1, zero_division=0)
    cola = (1 - config.CONFIANZA_BOOTSTRAP) / 2
    return {
        "accuracy_ci_baja": np.quantile(accs, cola), "accuracy_ci_alta": np.quantile(accs, 1 - cola),
        "f1_ofensivo_ci_baja": np.quantile(f1s, cola), "f1_ofensivo_ci_alta": np.quantile(f1s, 1 - cola),
    }


def predecir_baseline(pipeline, textos):
    return pipeline.predict(textos)


@torch.inference_mode()
def probabilidades_transformer(modelo, tokenizer, textos, tamano_lote=32):
    probs = []
    dispositivo = next(modelo.parameters()).device
    for i in range(0, len(textos), tamano_lote):
        lote = textos[i:i + tamano_lote]
        entradas = tokenizer(lote, return_tensors="pt", truncation=True, padding=True,
                             max_length=config.MAX_TOKENS).to(dispositivo)
        logits = modelo(**entradas).logits
        probs.extend(logits.softmax(-1)[:, 1].cpu().tolist())
    return np.array(probs)


def construir_predecir_ensamble(baseline, modelo, tokenizer, detectores_categoria):
    """Mismo criterio de votar.votar_texto (2 de 3 votos: baseline,
    transformer, comité de categorías), pero vectorizado sobre todo un
    conjunto de textos de una sola vez -- evaluar fila por fila con
    votar_texto sería correcto pero mucho más lento aquí (miles de filas)."""
    def predecir(textos):
        probs_baseline = np.asarray(baseline.predict_proba(textos))[:, 1]
        probs_transformer = probabilidades_transformer(modelo, tokenizer, textos)
        comite_categorias = np.zeros(len(textos), dtype=bool)
        for detector in detectores_categoria.values():
            comite_categorias |= np.asarray(detector.predict_proba(textos))[:, 1] >= config.UMBRAL_CATEGORIA

        votos = (probs_baseline >= 0.5).astype(int) + (probs_transformer >= 0.5).astype(int) + comite_categorias
        return (votos >= config.VOTOS_MINIMOS_OFENSIVO).astype(int)
    return predecir


def evaluar_modelo(nombre_modelo, predecir_fn, df, etiqueta_columna="ofensivo", con_bootstrap=False):
    predichas = predecir_fn(list(df["texto"]))
    fila_general = {"modelo": nombre_modelo, "subconjunto": "general", **_metricas(df[etiqueta_columna], predichas)}
    if con_bootstrap:
        fila_general.update(intervalo_confianza_bootstrap(df[etiqueta_columna], predichas))

    filas = [fila_general]
    for idioma, grupo in df.assign(_pred=predichas).groupby("idioma"):
        filas.append({"modelo": nombre_modelo, "subconjunto": f"idioma={idioma}",
                      **_metricas(grupo[etiqueta_columna], grupo["_pred"])})
    for fuente, grupo in df.assign(_pred=predichas).groupby("fuente"):
        filas.append({"modelo": nombre_modelo, "subconjunto": f"fuente={fuente}",
                      **_metricas(grupo[etiqueta_columna], grupo["_pred"])})
    return pd.DataFrame(filas), predichas


def main():
    print("Cargando prueba.tsv y prueba_codeswitch.tsv...")
    prueba = _leer_tsv(config.RUTA_PROCESADOS / "prueba.tsv")
    codeswitch = _leer_tsv(config.RUTA_PROCESADOS / "prueba_codeswitch.tsv")

    resultados = []
    baseline = modelo = tokenizer = detectores_categoria = None

    if config.RUTA_BASELINE.exists():
        print("\nEvaluando baseline (TF-IDF + Regresión Logística)...")
        baseline = joblib.load(config.RUTA_BASELINE)
        predecir_fn = lambda textos: predecir_baseline(baseline, textos)  # noqa: E731
        df_resultado, _ = evaluar_modelo("baseline", predecir_fn, prueba, con_bootstrap=True)
        resultados.append(df_resultado)
        df_resultado_cs, _ = evaluar_modelo("baseline", predecir_fn, codeswitch)
        df_resultado_cs["subconjunto"] = "codeswitch_" + df_resultado_cs["subconjunto"]
        resultados.append(df_resultado_cs)
    else:
        print(f"\nAviso: no hay baseline en {config.RUTA_BASELINE}, se omite "
              f"(corre primero: python entrenar_baseline.py)")

    if config.RUTA_MODELO_DIR.exists():
        print("\nEvaluando Transformer (XLM-R afinado)...")
        tokenizer = AutoTokenizer.from_pretrained(config.RUTA_MODELO_DIR)
        modelo = AutoModelForSequenceClassification.from_pretrained(config.RUTA_MODELO_DIR)
        modelo.eval()
        if torch.cuda.is_available():
            modelo.to("cuda")
        predecir_fn = lambda textos: (probabilidades_transformer(modelo, tokenizer, textos) >= 0.5).astype(int)  # noqa: E731

        df_resultado, pred_prueba_t = evaluar_modelo("xlm-roberta", predecir_fn, prueba, con_bootstrap=True)
        resultados.append(df_resultado)
        df_resultado_cs, _ = evaluar_modelo("xlm-roberta", predecir_fn, codeswitch)
        df_resultado_cs["subconjunto"] = "codeswitch_" + df_resultado_cs["subconjunto"]
        resultados.append(df_resultado_cs)

        matriz = confusion_matrix(prueba["ofensivo"], pred_prueba_t)
        pd.DataFrame(matriz, index=["real_no_ofensivo", "real_ofensivo"],
                     columns=["pred_no_ofensivo", "pred_ofensivo"]).to_csv(config.RUTA_MATRIZ_CONFUSION_CSV)
    else:
        print(f"\nAviso: no hay modelo afinado en {config.RUTA_MODELO_DIR}, se omite "
              f"(corre primero: python entrenar.py)")

    if config.RUTA_DETECTORES_CATEGORIA.exists():
        detectores_categoria = joblib.load(config.RUTA_DETECTORES_CATEGORIA)
    else:
        print(f"\nAviso: no hay detectores de categoría en {config.RUTA_DETECTORES_CATEGORIA}, se omite "
              f"(corre primero: python entrenar_categorias.py)")

    if baseline is not None and modelo is not None and detectores_categoria:
        print("\nEvaluando ensamble por votación (baseline + Transformer + comité de categorías)...")
        predecir_fn = construir_predecir_ensamble(baseline, modelo, tokenizer, detectores_categoria)

        df_resultado, pred_prueba_e = evaluar_modelo("voto_ensamble", predecir_fn, prueba, con_bootstrap=True)
        resultados.append(df_resultado)
        df_resultado_cs, _ = evaluar_modelo("voto_ensamble", predecir_fn, codeswitch)
        df_resultado_cs["subconjunto"] = "codeswitch_" + df_resultado_cs["subconjunto"]
        resultados.append(df_resultado_cs)

        matriz = confusion_matrix(prueba["ofensivo"], pred_prueba_e)
        pd.DataFrame(matriz, index=["real_no_ofensivo", "real_ofensivo"],
                     columns=["pred_no_ofensivo", "pred_ofensivo"]).to_csv(config.RUTA_MATRIZ_CONFUSION_ENSAMBLE_CSV)
    else:
        print("\nAviso: faltan piezas para el ensamble (baseline, Transformer y/o detectores de categoría), se omite.")

    if not resultados:
        raise SystemExit("No hay ningún modelo entrenado. Corre entrenar_baseline.py y/o entrenar.py primero.")

    tabla = pd.concat(resultados, ignore_index=True).round(4)
    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(config.RUTA_RESULTADOS_CSV, index=False)

    print("\n== Resultados generales (prueba.tsv), con intervalo de confianza del 95% ==")
    columnas = ["modelo", "subconjunto", "accuracy", "accuracy_ci_baja", "accuracy_ci_alta",
                "f1_ofensivo", "f1_ofensivo_ci_baja", "f1_ofensivo_ci_alta"]
    general = tabla[tabla["subconjunto"] == "general"]
    print(general[[c for c in columnas if c in general.columns]].to_string(index=False))
    print("\n== Desempeño con code-switching (prueba_codeswitch.tsv) ==")
    print(tabla[tabla["subconjunto"] == "codeswitch_general"].to_string(index=False))
    print(f"\nOK: tabla completa (por idioma y por fuente también) en {config.RUTA_RESULTADOS_CSV}")


if __name__ == "__main__":
    main()
