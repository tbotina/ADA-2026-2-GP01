"""Mide cuántos nodos visita el backtracking con poda frente al peor caso (árbol completo).

Tres experimentos, todos con semillas fijas:

1. La instancia base del enunciado, k = (4, 3, 3, 2, 2), con 100 semillas.
2. Crecimiento en m: k = (4, ..., 4) con m = 2, ..., 8, 30 semillas por tamaño.
3. Crecimiento en K: k = (K, ..., K) con m = 5 y K = 2, ..., 8, 30 semillas por tamaño.

El peor caso es N, el número de nodos del árbol completo (Proposición 2 del informe), que es
exponencial en m. Cada corrida guarda también el tiempo, para la comparación con T(n).

Al crecer m se ajusta la probabilidad por regla para que una configuración completa cubra, en
promedio, la misma fracción de reglas que en la instancia base: si no, con más categorías el
óptimo llega a n y la poda corta casi todo, y el experimento mediría instancias cada vez más
fáciles en lugar de más grandes. Al crecer K cada opción cubre lo mismo que en la base, así que
el óptimo mejora con K; es una propiedad real de tener más opciones, no un artefacto.

Uso, desde la raíz del repo (requiere matplotlib):

    python experimentos/medir_nodos.py

Salida: experimentos/resultados/nodos_poda.csv y figuras/nodos_poda.png.
"""
import csv
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from simulador.exacto import backtracking, nodos_peor_caso  # noqa: E402
from simulador.modelo import N_BASE, PROB_REGLA_BASE, RHO_BASE, generar, instancia_base  # noqa: E402

CSV = RAIZ / "experimentos" / "resultados" / "nodos_poda.csv"
FIGURA = RAIZ / "figuras" / "nodos_poda.png"
SEMILLAS_BASE = 100
SEMILLAS = 30
EMES = range(2, 9)
KAS = range(2, 9)
# Fracción esperada de reglas que cubre una configuración completa en la base (m = 5).
DENSIDAD_BASE = 1 - (1 - PROB_REGLA_BASE) ** 5


def prob_regla_para(m):
    """Probabilidad por regla con la que m categorías cubren la misma fracción que la base."""
    return 1 - (1 - DENSIDAD_BASE) ** (1 / m)


def correr(nombre, instancia, semilla, prob_regla):
    t0 = time.perf_counter()
    r = backtracking(instancia)
    segundos = time.perf_counter() - t0
    return {
        "experimento": nombre, "m": instancia.m, "K": instancia.K, "n": instancia.n, "rho": instancia.rho,
        "prob_regla": round(prob_regla, 4), "semilla": semilla, "configuraciones": instancia.tamano_espacio,
        "peor_caso": nodos_peor_caso(instancia.k), "nodos": r.nodos, "podas": r.podas,
        "aciertos": r.aciertos, "segundos": round(segundos, 6),
    }


def medir():
    filas = []
    for semilla in range(SEMILLAS_BASE):
        filas.append(correr("base", instancia_base(semilla), semilla, PROB_REGLA_BASE))
    for m in EMES:
        for semilla in range(SEMILLAS):
            instancia = generar((4,) * m, N_BASE, RHO_BASE, 1000 * m + semilla, prob_regla_para(m))
            filas.append(correr("crece_m", instancia, semilla, prob_regla_para(m)))
    for K in KAS:
        for semilla in range(SEMILLAS):
            instancia = generar((K,) * 5, N_BASE, RHO_BASE, 2000 * K + semilla, PROB_REGLA_BASE)
            filas.append(correr("crece_K", instancia, semilla, PROB_REGLA_BASE))
    return filas


def resumir(filas, experimento, clave):
    """Por valor de `clave`: mediana, mínimo y máximo de nodos, y el peor caso."""
    grupos = {}
    for f in filas:
        if f["experimento"] == experimento:
            grupos.setdefault(f[clave], []).append(f)
    resumen = []
    for valor, grupo in sorted(grupos.items()):
        nodos = [g["nodos"] for g in grupo]
        resumen.append({
            clave: valor, "peor_caso": grupo[0]["peor_caso"], "configuraciones": grupo[0]["configuraciones"],
            "mediana": statistics.median(nodos), "minimo": min(nodos), "maximo": max(nodos),
            "fraccion": statistics.median(nodos) / grupo[0]["peor_caso"],
            "segundos": statistics.median(g["segundos"] for g in grupo),
            "aciertos": statistics.median(g["aciertos"] for g in grupo),
        })
    return resumen


def dibujar(filas):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    AZUL, GRIS, NARANJA, TINTA, TINTA_2, REJILLA = "#2a78d6", "#9a9893", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
    fig, ejes = plt.subplots(1, 2, figsize=(8.5, 3.6), dpi=200)
    fig.patch.set_facecolor("white")

    for ax, (experimento, clave, etiqueta) in zip(ejes, [("crece_m", "m", "Número de categorías $m$ ($K = 4$)"),
                                                        ("crece_K", "K", "Opciones por categoría $K$ ($m = 5$)")]):
        resumen = resumir(filas, experimento, clave)
        x = [r[clave] for r in resumen]
        ax.plot(x, [r["peor_caso"] for r in resumen], color=GRIS, lw=2, label="Peor caso: árbol completo, $N$")
        ax.plot(x, [r["configuraciones"] for r in resumen], color=NARANJA, lw=1.5, ls="--", label="Configuraciones, $|C|$")
        ax.fill_between(x, [r["minimo"] for r in resumen], [r["maximo"] for r in resumen], color=AZUL, alpha=0.15, lw=0)
        ax.plot(x, [r["mediana"] for r in resumen], color=AZUL, lw=2, marker="o", ms=4,
                label="Nodos visitados con poda (mediana; banda: mín–máx)")
        ax.set_yscale("log")
        ax.set_xlabel(etiqueta, color=TINTA)
        ax.set_xticks(list(x))
        ax.grid(axis="y", color=REJILLA, lw=1)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(TINTA_2)
        ax.tick_params(colors=TINTA_2)
    ejes[0].set_ylabel("Nodos (escala logarítmica)", color=TINTA)
    manijas, etiquetas = ejes[0].get_legend_handles_labels()
    fig.legend(manijas, etiquetas, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    FIGURA.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURA, facecolor="white")


def main():
    filas = medir()
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    print("datos:", CSV)

    base = [f["nodos"] for f in filas if f["experimento"] == "base"]
    print(f"\nInstancia base, {len(base)} semillas: peor caso {nodos_peor_caso((4, 3, 3, 2, 2))} nodos; "
          f"con poda mediana {statistics.median(base):.0f}, mínimo {min(base)}, máximo {max(base)}, "
          f"media {statistics.mean(base):.1f}")
    for experimento, clave in [("crece_m", "m"), ("crece_K", "K")]:
        print(f"\n{clave:>2}  {'|C|':>7} {'peor N':>7} {'mediana':>8} {'mín':>7} {'máx':>7}  fracción  "
              f"óptimo/{N_BASE}  seg")
        for r in resumir(filas, experimento, clave):
            print(f"{r[clave]:>2}  {r['configuraciones']:>7} {r['peor_caso']:>7} {r['mediana']:>8.0f} "
                  f"{r['minimo']:>7} {r['maximo']:>7}  {r['fraccion']:>7.3f}  {r['aciertos']:>9.0f}  "
                  f"{r['segundos']:.3f}")
    dibujar(filas)
    print("\nfigura:", FIGURA)


if __name__ == "__main__":
    main()
