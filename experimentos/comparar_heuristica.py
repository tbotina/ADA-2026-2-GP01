"""Compara la búsqueda local por coordenadas con el exacto, sobre el simulador y con semillas fijas.

1. Instancia base: el voraz desde ninguna opción, la búsqueda local desde ninguna opción, y la
   búsqueda local desde cada una de las 144 configuraciones.
2. Teorema 2.3.1: 300 instancias con una regla por instancia (|R_i| = 1), razón voraz / óptimo.
3. Caso general: 300 instancias con 2 o 3 reglas por instancia, razón local / óptimo desde ninguna
   opción y desde 5 inicios al azar (se conserva el mejor).
4. Contraejemplo generalizado: razón 1/q para varios q y m.

Uso, desde la raíz del repo:

    python experimentos/comparar_heuristica.py

Salida: experimentos/resultados/heuristica_vs_exacto.csv y un resumen en pantalla.
"""
import csv
import random
import statistics
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from simulador.exacto import backtracking  # noqa: E402
from simulador.heuristica import busqueda_local, voraz  # noqa: E402
from simulador.instancias import contraejemplo  # noqa: E402
from simulador.modelo import aciertos, configuraciones, generar, instancia_base  # noqa: E402

CSV = RAIZ / "experimentos" / "resultados" / "heuristica_vs_exacto.csv"
CUANTAS = 300
REINICIOS = 5


def al_azar(azar, reglas_por_instancia):
    k = tuple(azar.randint(2, 5) for _ in range(azar.randint(2, 6)))
    return generar(k, azar.randint(20, 150), azar.randint(10, 30), azar.randrange(10**6),
                   prob_regla=azar.choice((0.1, 0.2, 0.3)), reglas_por_instancia=reglas_por_instancia)


def fila(experimento, instancia, semilla, metodo, r, optimo):
    return {"experimento": experimento, "semilla": semilla, "m": instancia.m, "K": instancia.K,
            "n": instancia.n, "metodo": metodo, "aciertos": r.aciertos, "optimo": optimo.aciertos,
            "razon": round(r.aciertos / optimo.aciertos, 4) if optimo.aciertos else 1.0,
            "pasadas": r.pasadas, "evaluaciones": r.evaluaciones, "nodos_exacto": optimo.nodos}


def main():
    filas = []

    base = instancia_base()
    optimo = backtracking(base)
    filas.append(fila("base", base, 0, "voraz", voraz(base), optimo))
    filas.append(fila("base", base, 0, "local_desde_vacio", busqueda_local(base), optimo))
    desde_todas = [busqueda_local(base, c) for c in configuraciones(base)]
    for c, r in zip(configuraciones(base), desde_todas):
        filas.append(fila("base_144_inicios", base, c, "local", r, optimo))

    azar = random.Random(2026)
    for semilla in range(CUANTAS):
        instancia = al_azar(azar, (1, 1))
        filas.append(fila("una_regla", instancia, semilla, "voraz", voraz(instancia), backtracking(instancia)))
    for semilla in range(CUANTAS):
        instancia = al_azar(azar, (2, 3))
        optimo = backtracking(instancia)
        filas.append(fila("varias_reglas", instancia, semilla, "local_desde_vacio", busqueda_local(instancia), optimo))
        mejor = None
        for _ in range(REINICIOS):
            inicio = tuple(azar.randrange(kj) for kj in instancia.k)
            r = busqueda_local(instancia, inicio)
            if mejor is None or r.aciertos > mejor.aciertos:
                mejor = mejor._replace(evaluaciones=mejor.evaluaciones + r.evaluaciones) if mejor else r
            else:
                mejor = mejor._replace(evaluaciones=mejor.evaluaciones + r.evaluaciones)
        filas.append(fila("varias_reglas", instancia, semilla, f"local_{REINICIOS}_reinicios", mejor, optimo))

    for q in (2, 10, 100, 1000):
        for m in (2, 3, 5):
            instancia = contraejemplo(q, m)
            r = busqueda_local(instancia, (0,) * m)
            filas.append(fila("contraejemplo", instancia, f"q={q},m={m}", "local_desde_ceros", r, backtracking(instancia)))

    CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    print("datos:", CSV, "\n")

    def resumen(experimento, metodo):
        g = [x for x in filas if x["experimento"] == experimento and x["metodo"] == metodo]
        razones = [x["razon"] for x in g]
        optimos = sum(x["aciertos"] == x["optimo"] for x in g)
        evals = statistics.median(x["evaluaciones"] for x in g)
        nodos = statistics.median(x["nodos_exacto"] for x in g)
        return (f"{experimento:>16} {metodo:>18}: {len(g):>3} casos, razón mín {min(razones):.3f}, "
                f"mediana {statistics.median(razones):.3f}, media {statistics.mean(razones):.3f}; "
                f"óptimo en {optimos} ({100 * optimos / len(g):.0f} %); evals mediana {evals:.0f} "
                f"frente a {nodos:.0f} nodos del exacto")

    b = [x for x in filas if x["experimento"] == "base"]
    print(f"Instancia base: óptimo {b[0]['optimo']}; voraz {b[0]['aciertos']} ({b[0]['evaluaciones']} evals); "
          f"local desde vacío {b[1]['aciertos']} ({b[1]['pasadas']} pasadas, {b[1]['evaluaciones']} evals)")
    print(resumen("base_144_inicios", "local"))
    print(resumen("una_regla", "voraz"))
    print(resumen("varias_reglas", "local_desde_vacio"))
    print(resumen("varias_reglas", f"local_{REINICIOS}_reinicios"))
    print("contraejemplo:", ", ".join(f"{x['semilla']} → {x['aciertos']}/{x['optimo']}"
                                      for x in filas if x["experimento"] == "contraejemplo"))


if __name__ == "__main__":
    main()
