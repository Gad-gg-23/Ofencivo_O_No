"""
Descarga las 4 fuentes de config.FUENTES desde Hugging Face Datasets y las
deja, cada una por separado, en data/crudos/<fuente>.csv con el mismo
esquema: texto, idioma, ofensivo (0/1), fuente, split_original, más una
columna extra_<columna> por cada entrada en "columnas_extra" de esa fuente
(piuba_odio: RACISM/CLASS/etc.; OffendES: la etiqueta NO/NOE/OFP/OFG
original) -- esas alimentan a entrenar_categorias.py.

Uso:
    python descargar_datos.py                     # las 4 fuentes
    python descargar_datos.py --fuente offendes    # solo una

OffendES es GATED: hay que entrar a
https://huggingface.co/datasets/SINAI/OffendES, iniciar sesión y aceptar las
condiciones ANTES de correr este script, y haber hecho `hf auth login` (o
tener la variable de entorno HF_TOKEN) -- ver GUIA_EQUIPO.md. Las otras 3
fuentes son públicas y no necesitan ni cuenta ni token.
"""

import argparse
import sys

import pandas as pd

import config


def _resolver_etiqueta(ejemplo, columna_label, feature):
    """Convierte el valor crudo de la etiqueta a su nombre (string), para
    que el mapeo en config.FUENTES no dependa de qué entero le tocó a cada
    clase en esta versión del dataset -- solo del nombre de la clase."""
    valor = ejemplo[columna_label]
    if hasattr(feature, "int2str"):
        return feature.int2str(valor)
    return str(valor)


def descargar_fuente(nombre):
    from datasets import load_dataset

    info = config.FUENTES[nombre]
    sufijo_config = f" (config={info['config']})" if info["config"] else ""
    print(f"{nombre}: descargando {info['repo']}{sufijo_config}...")

    try:
        ds_dict = load_dataset(info["repo"], info["config"]) if info["config"] else load_dataset(info["repo"])
    except Exception as error:
        extra = (" Revisa que hayas aceptado las condiciones del dataset en "
                  "huggingface.co y que hayas hecho `hf auth login`." if info["gated"] else "")
        sys.exit(f"Error descargando {nombre} ({info['repo']}): {error}.{extra}")

    columna_texto, columna_label = info["columna_texto"], info["columna_label"]
    columnas_extra = info.get("columnas_extra", [])
    filas = []
    for split in info["splits"]:
        if split not in ds_dict:
            continue
        ds = ds_dict[split]
        feature = ds.features[columna_label]
        features_extra = {col: ds.features[col] for col in columnas_extra}
        for ejemplo in ds:
            texto = ejemplo[columna_texto]
            etiqueta_nombre = _resolver_etiqueta(ejemplo, columna_label, feature)
            if etiqueta_nombre not in info["mapa_label"]:
                continue  # etiqueta inesperada (no debería pasar, pero no se descarta todo el dataset por eso)
            fila = {
                "texto": texto,
                "idioma": info["idioma"],
                "ofensivo": info["mapa_label"][etiqueta_nombre],
                "fuente": nombre,
                "split_original": split,
            }
            for col in columnas_extra:
                fila[f"extra_{col}"] = _resolver_etiqueta(ejemplo, col, features_extra[col])
            filas.append(fila)

    df = pd.DataFrame(filas)
    config.RUTA_CRUDOS.mkdir(parents=True, exist_ok=True)
    ruta_salida = config.RUTA_CRUDOS / f"{nombre}.csv"
    df.to_csv(ruta_salida, index=False)
    print(f"  OK: {len(df):,} filas -> {ruta_salida} "
          f"({df['ofensivo'].mean():.1%} ofensivo)")


def main():
    parser = argparse.ArgumentParser(description="Descarga las fuentes de datos desde Hugging Face.")
    parser.add_argument("--fuente", choices=list(config.FUENTES), help="Descargar solo esta fuente.")
    args = parser.parse_args()

    fuentes_elegidas = [args.fuente] if args.fuente else list(config.FUENTES)
    for nombre in fuentes_elegidas:
        descargar_fuente(nombre)

    print(f"\nOK: fuentes crudas en {config.RUTA_CRUDOS}")


if __name__ == "__main__":
    main()
