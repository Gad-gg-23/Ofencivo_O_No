"""
Evalúa el baseline (TF-IDF + Regresión Logística) y el Transformer (XLM-R
afinado) sobre prueba.tsv y prueba_codeswitch.tsv, ninguno de los dos visto
en entrenamiento.

    python evaluar.py

Tres cosas, todas guardadas en salidas/:

1. Comparación baseline vs. Transformer sobre prueba.tsv, con Accuracy,
   Precision, Recall y F1 (estándar de clasificación binaria) -- para
   justificar que el Transformer vale la inversión de tiempo de GPU.
2. Desglose por idioma (es/en) y por fuente, para ver si el modelo
   generaliza igual a los 4 datasets de origen o solo a algunos.
3. Desempeño sobre prueba_codeswitch.tsv (frases mixtas es+en) -- la parte
   que justifica haber elegido un modelo multilingüe en vez de uno por
   idioma.
"""

import csv

import joblib
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
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


def predecir_baseline(pipeline, textos):
    return pipeline.predict(textos)


@torch.inference_mode()
def predecir_transformer(modelo, tokenizer, textos, tamano_lote=32):
    predichas = []
    dispositivo = next(modelo.parameters()).device
    for i in range(0, len(textos), tamano_lote):
        lote = textos[i:i + tamano_lote]
        entradas = tokenizer(lote, return_tensors="pt", truncation=True, padding=True,
                             max_length=config.MAX_TOKENS).to(dispositivo)
        logits = modelo(**entradas).logits
        predichas.extend(logits.argmax(-1).cpu().tolist())
    return predichas


def evaluar_modelo(nombre_modelo, predecir_fn, df, etiqueta_columna="ofensivo"):
    predichas = predecir_fn(list(df["texto"]))
    fila_general = {"modelo": nombre_modelo, "subconjunto": "general", **_metricas(df[etiqueta_columna], predichas)}

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

    if config.RUTA_BASELINE.exists():
        print("\nEvaluando baseline (TF-IDF + Regresión Logística)...")
        baseline = joblib.load(config.RUTA_BASELINE)
        predecir_fn = lambda textos: predecir_baseline(baseline, textos)  # noqa: E731
        df_resultado, pred_prueba = evaluar_modelo("baseline", predecir_fn, prueba)
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
        predecir_fn = lambda textos: predecir_transformer(modelo, tokenizer, textos)  # noqa: E731

        df_resultado, pred_prueba_t = evaluar_modelo("xlm-roberta", predecir_fn, prueba)
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

    if not resultados:
        raise SystemExit("No hay ningún modelo entrenado. Corre entrenar_baseline.py y/o entrenar.py primero.")

    tabla = pd.concat(resultados, ignore_index=True).round(4)
    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(config.RUTA_RESULTADOS_CSV, index=False)

    print("\n== Resultados generales (prueba.tsv) ==")
    print(tabla[tabla["subconjunto"] == "general"].to_string(index=False))
    print("\n== Desempeño con code-switching (prueba_codeswitch.tsv) ==")
    print(tabla[tabla["subconjunto"] == "codeswitch_general"].to_string(index=False))
    print(f"\nOK: tabla completa (por idioma y por fuente también) en {config.RUTA_RESULTADOS_CSV}")


if __name__ == "__main__":
    main()
