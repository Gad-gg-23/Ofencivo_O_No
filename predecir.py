"""
Clasifica una frase como ofensiva o no ofensiva con el Transformer ya
afinado. Busca, en orden: el modelo local (salidas/modelo_xlmr/, el que deja
entrenar.py), el modelo compartido del equipo en Hugging Face
(config.HF_REPO_MODELO, se descarga solo la primera vez y queda en caché), y
por último xlm-roberta-base sin afinar (con aviso de que las predicciones no
van a ser confiables).

    python predecir.py --texto "esta frase es una prueba"

Con --explicar, además de la predicción, quita una palabra a la vez y mide
cuánto cae la probabilidad de "ofensivo" al quitarla -- la(s) palabra(s) con
mayor caída son las que más "convencieron" al modelo de que la frase es
ofensiva (leave-one-out, una forma simple de interpretabilidad sin
necesitar librerías extra):

    python predecir.py --texto "esta frase es una prueba" --explicar
"""

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

_modelo = None
_tokenizer = None


def cargar_modelo():
    global _modelo, _tokenizer
    if _modelo is None:
        if config.RUTA_MODELO_DIR.exists():
            ruta = config.RUTA_MODELO_DIR
        else:
            ruta = config.HF_REPO_MODELO
            print(f"Aviso: no hay modelo afinado local en {config.RUTA_MODELO_DIR}. "
                  f"Descargando el del equipo desde https://huggingface.co/{config.HF_REPO_MODELO} "
                  f"(solo la primera vez, luego queda en caché)...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(ruta)
            _modelo = AutoModelForSequenceClassification.from_pretrained(ruta)
        except Exception as error:
            print(f"Aviso: no se pudo cargar {ruta} ({error}). Usando {config.MODELO_BASE} sin afinar "
                  f"-- las predicciones NO son confiables.")
            _tokenizer = AutoTokenizer.from_pretrained(config.MODELO_BASE)
            _modelo = AutoModelForSequenceClassification.from_pretrained(config.MODELO_BASE, num_labels=2)
        _modelo.eval()
        if torch.cuda.is_available():
            _modelo.to("cuda")
    return _modelo, _tokenizer


@torch.inference_mode()
def probabilidad_ofensivo(texto):
    modelo, tokenizer = cargar_modelo()
    entradas = tokenizer(texto, return_tensors="pt", truncation=True, max_length=config.MAX_TOKENS)
    entradas = {k: v.to(modelo.device) for k, v in entradas.items()}
    probabilidades = modelo(**entradas).logits.softmax(-1)[0]
    return probabilidades[1].item()  # índice 1 = ofensivo (ver config.ETIQUETAS)


def clasificar(texto):
    prob = probabilidad_ofensivo(texto)
    return config.ETIQUETAS[int(prob >= 0.5)], prob


def explicar(texto):
    """Leave-one-out: quita cada palabra, una a la vez, y mide cuánto cae
    probabilidad_ofensivo(texto) al quitarla. Una caída grande significa que
    esa palabra pesaba mucho en la predicción."""
    palabras = texto.split()
    prob_completa = probabilidad_ofensivo(texto)
    aportes = []
    for i in range(len(palabras)):
        texto_sin_palabra = " ".join(palabras[:i] + palabras[i + 1:])
        if not texto_sin_palabra:
            continue
        prob_sin = probabilidad_ofensivo(texto_sin_palabra)
        aportes.append((palabras[i], prob_completa - prob_sin))
    return sorted(aportes, key=lambda par: par[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Clasifica una frase como ofensiva o no ofensiva.")
    parser.add_argument("--texto", required=True)
    parser.add_argument("--explicar", action="store_true",
                        help="Muestra qué palabras más influyeron en la predicción (leave-one-out).")
    args = parser.parse_args()

    etiqueta, prob = clasificar(args.texto)
    print(f"{etiqueta}  (probabilidad de ofensivo: {prob:.1%})")

    if args.explicar:
        print("\nPalabras que más empujan hacia 'ofensivo' (quitarlas bajó más la probabilidad):")
        for palabra, aporte in explicar(args.texto)[:5]:
            signo = "+" if aporte >= 0 else ""
            print(f"  {palabra!r}: {signo}{aporte:.1%}")


if __name__ == "__main__":
    main()
