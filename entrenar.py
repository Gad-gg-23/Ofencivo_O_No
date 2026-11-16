"""
Afina XLM-RoBERTa (xlm-roberta-base) sobre entrenamiento.tsv/validacion.tsv
ya preparados.

    python entrenar.py

Requiere haber corrido antes `python preparar_datos.py`. Pensado para correr
en Colab con GPU: en CPU tardaría horas en vez de minutos (ver
colab/Ofensivo_Colab.ipynb).

Las clases están desbalanceadas (hay más "no ofensivo" que "ofensivo" en casi
todas las fuentes), así que se usa una CrossEntropyLoss con pesos inversos a
la frecuencia de cada clase, en vez de dejar que el modelo aprenda a
predecir siempre "no ofensivo" y aun así tener accuracy alto.
"""

import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import config


def _leer_tsv(ruta):
    return pd.read_csv(ruta, sep="\t", quoting=csv.QUOTE_NONE, escapechar="\\", keep_default_na=False)


def tokenizar_dataset(df, tokenizer):
    ds = Dataset.from_dict({"text": list(df["texto"]), "label": list(df["ofensivo"])})
    return ds.map(
        lambda lote: tokenizer(lote["text"], truncation=True, max_length=config.MAX_TOKENS),
        batched=True,
    )


def calcular_pesos_clase(etiquetas):
    """Peso inverso a la frecuencia: la clase minoritaria pesa más en la
    pérdida, para que el modelo no gane accuracy solo por predecir siempre
    la clase mayoritaria."""
    conteos = np.bincount(etiquetas, minlength=2)
    pesos = len(etiquetas) / (2 * conteos)
    return torch.tensor(pesos, dtype=torch.float)


class EntrenadorConPesos(Trainer):
    def __init__(self, pesos_clase, **kwargs):
        super().__init__(**kwargs)
        self.pesos_clase = pesos_clase

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        etiquetas = inputs.pop("labels")
        salidas = model(**inputs)
        perdida = nn.functional.cross_entropy(
            salidas.logits, etiquetas, weight=self.pesos_clase.to(salidas.logits.device)
        )
        return (perdida, salidas) if return_outputs else perdida


def metricas(pred):
    etiquetas = pred.label_ids
    predichas = pred.predictions.argmax(-1)
    return {
        "f1_macro": f1_score(etiquetas, predichas, average="macro"),
        "f1_ofensivo": f1_score(etiquetas, predichas, pos_label=1),
    }


def entrenar():
    print(f"Cargando {config.MODELO_BASE} (preentrenado)...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODELO_BASE)
    modelo = AutoModelForSequenceClassification.from_pretrained(config.MODELO_BASE, num_labels=2)

    print("Tokenizando datos...")
    train = _leer_tsv(config.RUTA_PROCESADOS / "entrenamiento.tsv")
    val = _leer_tsv(config.RUTA_PROCESADOS / "validacion.tsv")
    ds_train = tokenizar_dataset(train, tokenizer)
    ds_val = tokenizar_dataset(val, tokenizer)

    pesos_clase = calcular_pesos_clase(train["ofensivo"].to_numpy())
    print(f"  Pesos de clase (no_ofensivo, ofensivo): {pesos_clase.tolist()}")

    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    ruta_checkpoints = config.RUTA_SALIDAS / "checkpoints_temporales"
    argumentos = TrainingArguments(
        output_dir=str(ruta_checkpoints),
        num_train_epochs=config.EPOCAS,
        per_device_train_batch_size=config.TAMANO_LOTE,
        per_device_eval_batch_size=config.TAMANO_LOTE,
        gradient_accumulation_steps=config.ACUMULACION_GRADIENTE,
        learning_rate=config.TASA_APRENDIZAJE,
        warmup_steps=config.PASOS_CALENTAMIENTO,
        eval_strategy="epoch",
        save_strategy="no",  # el único modelo que importa es el final afinado
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to=[],
        seed=config.SEMILLA,
    )

    entrenador = EntrenadorConPesos(
        pesos_clase=pesos_clase,
        model=modelo,
        args=argumentos,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        compute_metrics=metricas,
    )

    print("Afinando (esto tarda varios minutos, va a imprimir el progreso por época)...")
    resultado = entrenador.train()

    pd.DataFrame(entrenador.state.log_history).to_csv(config.RUTA_HISTORICO_CSV, index=False)

    modelo.save_pretrained(config.RUTA_MODELO_DIR)
    tokenizer.save_pretrained(config.RUTA_MODELO_DIR)
    import shutil
    shutil.rmtree(ruta_checkpoints, ignore_errors=True)

    print(f"OK: modelo afinado en {config.RUTA_MODELO_DIR}")
    print(f"    pérdida final de entrenamiento: {resultado.training_loss:.4f}")


if __name__ == "__main__":
    entrenar()
