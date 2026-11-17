"""
Ensamble por votación: combina 3 "votantes" independientes para decidir si
una frase es ofensiva, en vez de confiar en un solo modelo:

1. El baseline general (TF-IDF + Regresión Logística, entrenar_baseline.py).
2. El Transformer general (XLM-RoBERTa afinado, entrenar.py).
3. El "comité de categorías": vota ofensivo si CUALQUIER detector de
   config.CATEGORIAS se activa (racismo, clasismo, insulto personal, etc.,
   entrenados por entrenar_categorias.py) -- cuenta como UN solo voto, no
   uno por categoría, para no sesgar la votación hacia "ofensivo" solo por
   tener varios detectores.

Ofensivo si al menos config.VOTOS_MINIMOS_OFENSIVO de los 3 votan ofensivo
(mayoría simple con VOTOS_MINIMOS_OFENSIVO=2, ver config.py).

Requiere haber corrido antes entrenar_baseline.py, entrenar.py y
entrenar_categorias.py (los tres, si no faltan votantes -- ver
componentes_disponibles()).
"""

import joblib

import config
import predecir

_baseline = None
_detectores_categoria = None


def _cargar_baseline():
    global _baseline
    if _baseline is None and config.RUTA_BASELINE.exists():
        _baseline = joblib.load(config.RUTA_BASELINE)
    return _baseline


def _cargar_detectores_categoria():
    global _detectores_categoria
    if _detectores_categoria is None and config.RUTA_DETECTORES_CATEGORIA.exists():
        _detectores_categoria = joblib.load(config.RUTA_DETECTORES_CATEGORIA)
    return _detectores_categoria


def _hay_transformer():
    """True si hay un modelo afinado local, O si el del equipo existe en
    Hugging Face (predecir.cargar_modelo() lo descarga solo). Sin este
    segundo caso, el ensamble se reportaría como "no disponible" en
    cualquier máquina que no haya corrido entrenar.py, aunque el modelo del
    equipo ya esté compartido -- justo el caso de usar el ensamble sin
    haber entrenado el Transformer uno mismo."""
    if config.RUTA_MODELO_DIR.exists():
        return True
    try:
        from huggingface_hub import repo_exists
        return repo_exists(config.HF_REPO_MODELO)
    except Exception:
        return False


def componentes_disponibles():
    """Qué votantes se pueden calcular ahora mismo, para que evaluar.py y
    predecir.py decidan si corren el ensamble completo o avisan qué falta."""
    return {
        "baseline": _cargar_baseline() is not None,
        "transformer": _hay_transformer(),
        "categorias": bool(_cargar_detectores_categoria()),
    }


def probabilidad_baseline(texto):
    baseline = _cargar_baseline()
    if baseline is None:
        raise SystemExit(f"No hay baseline en {config.RUTA_BASELINE}. Corre primero: python entrenar_baseline.py")
    return baseline.predict_proba([texto])[0][1]


def probabilidades_categoria(texto):
    """{nombre_categoria: probabilidad}, una por cada detector entrenado."""
    detectores = _cargar_detectores_categoria()
    if not detectores:
        raise SystemExit(f"No hay detectores de categoría en {config.RUTA_DETECTORES_CATEGORIA}. "
                          f"Corre primero: python entrenar_categorias.py")
    return {nombre: modelo.predict_proba([texto])[0][1] for nombre, modelo in detectores.items()}


def votar_texto(texto):
    """Devuelve (etiqueta_final, detalle). detalle trae la probabilidad de
    cada votante y qué categorías específicas se activaron, para poder
    explicar la decisión (ej. "ofensivo por: Transformer + comité de
    categorías [racismo, insulto_personal]")."""
    prob_baseline = probabilidad_baseline(texto)
    prob_transformer = predecir.probabilidad_ofensivo(texto)
    probs_categoria = probabilidades_categoria(texto)
    categorias_activas = [nombre for nombre, prob in probs_categoria.items() if prob >= config.UMBRAL_CATEGORIA]

    votos = {
        "baseline": prob_baseline >= 0.5,
        "transformer": prob_transformer >= 0.5,
        "comite_categorias": len(categorias_activas) > 0,
    }
    total_votos = sum(votos.values())
    etiqueta = config.ETIQUETAS[int(total_votos >= config.VOTOS_MINIMOS_OFENSIVO)]

    detalle = {
        "prob_baseline": prob_baseline,
        "prob_transformer": prob_transformer,
        "probs_categoria": probs_categoria,
        "categorias_activas": categorias_activas,
        "votos": votos,
        "total_votos": total_votos,
    }
    return etiqueta, detalle


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clasifica una frase con el ensamble por votación.")
    parser.add_argument("--texto", required=True)
    args = parser.parse_args()

    faltantes = [nombre for nombre, ok in componentes_disponibles().items() if not ok]
    if faltantes:
        raise SystemExit(f"Faltan estos componentes para votar: {faltantes}. "
                          f"Corre entrenar_baseline.py / entrenar.py / entrenar_categorias.py según falte.")

    etiqueta, detalle = votar_texto(args.texto)
    print(f"{etiqueta}  ({detalle['total_votos']}/3 votos)")
    print(f"  baseline:     {'ofensivo' if detalle['votos']['baseline'] else 'no_ofensivo'} "
          f"(prob={detalle['prob_baseline']:.1%})")
    print(f"  transformer:  {'ofensivo' if detalle['votos']['transformer'] else 'no_ofensivo'} "
          f"(prob={detalle['prob_transformer']:.1%})")
    comite = detalle["categorias_activas"] or ["ninguna"]
    print(f"  comité categorías: {'ofensivo' if detalle['votos']['comite_categorias'] else 'no_ofensivo'} "
          f"(categorías activas: {', '.join(comite)})")


if __name__ == "__main__":
    main()
