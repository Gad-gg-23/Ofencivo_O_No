"""
Punto de entrada único del clasificador Ofensivo / No Ofensivo (es + en).

    python Ofensivo_o_NoOfensivo.py descargar          -> baja las 4 fuentes desde Hugging Face
    python Ofensivo_o_NoOfensivo.py preparar           -> limpia, une y arma train/val/test + codeswitch
    python Ofensivo_o_NoOfensivo.py entrenar-base      -> baseline TF-IDF + Regresión Logística (con K-Fold)
    python Ofensivo_o_NoOfensivo.py entrenar           -> afina XLM-RoBERTa sobre los datos preparados
    python Ofensivo_o_NoOfensivo.py entrenar-categorias -> detectores de racismo/clasismo/etc.
    python Ofensivo_o_NoOfensivo.py evaluar            -> compara baseline vs. Transformer vs. ensamble
    python Ofensivo_o_NoOfensivo.py votar --texto "..."       -> clasifica con el ensamble por votación
    python Ofensivo_o_NoOfensivo.py predecir --texto "..." [--explicar]
"""

import argparse
import sys


def comando_descargar(args_restantes):
    import descargar_datos

    sys.argv = [sys.argv[0]] + args_restantes
    descargar_datos.main()


def comando_preparar():
    import preparar_datos

    preparar_datos.main()


def comando_entrenar_base():
    import entrenar_baseline

    entrenar_baseline.main()


def comando_entrenar():
    import entrenar

    entrenar.entrenar()


def comando_entrenar_categorias():
    import entrenar_categorias

    entrenar_categorias.main()


def comando_evaluar():
    import evaluar

    evaluar.main()


def comando_votar(args_restantes):
    import votar

    sys.argv = [sys.argv[0]] + args_restantes
    votar.main()


def comando_predecir(args_restantes):
    import predecir

    sys.argv = [sys.argv[0]] + args_restantes
    predecir.main()


def main():
    parser = argparse.ArgumentParser(description="Clasificador Ofensivo / No Ofensivo (español + inglés).")
    subparsers = parser.add_subparsers(dest="comando", required=True)
    subparsers.add_parser("descargar", help="Descarga las fuentes desde Hugging Face.")
    subparsers.add_parser("preparar", help="Limpia, une y arma los conjuntos (+ code-switching).")
    subparsers.add_parser("entrenar-base", help="Entrena el baseline TF-IDF + Regresión Logística (con K-Fold).")
    subparsers.add_parser("entrenar", help="Afina XLM-RoBERTa sobre los datos preparados.")
    subparsers.add_parser("entrenar-categorias", help="Entrena los detectores de racismo/clasismo/etc.")
    subparsers.add_parser("evaluar", help="Compara baseline vs. Transformer vs. ensamble (incluye code-switching).")
    subparsers.add_parser("votar", help="Clasifica una frase con el ensamble por votación.")
    subparsers.add_parser("predecir", help="Clasifica una frase como ofensiva o no ofensiva.")

    args, restantes = parser.parse_known_args()

    if args.comando == "descargar":
        comando_descargar(restantes)
    elif args.comando == "preparar":
        comando_preparar()
    elif args.comando == "entrenar-base":
        comando_entrenar_base()
    elif args.comando == "entrenar":
        comando_entrenar()
    elif args.comando == "entrenar-categorias":
        comando_entrenar_categorias()
    elif args.comando == "evaluar":
        comando_evaluar()
    elif args.comando == "votar":
        comando_votar(restantes)
    elif args.comando == "predecir":
        comando_predecir(restantes)


if __name__ == "__main__":
    main()
