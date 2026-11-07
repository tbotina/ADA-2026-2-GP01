"""Dibuja la curva de la búsqueda a partir de notebooks/registro_evaluaciones.csv.

Muestra, por número de evaluación, los aciertos de cada configuración (puntos grises)
y el mejor resultado hasta ese momento (escalón azul), con las etapas sombreadas.

Uso, desde la raíz del repo (requiere matplotlib):

    python figuras/generar_curva.py

Salida: figuras/curva_busqueda.png
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
REGISTRO = RAIZ / "notebooks" / "registro_evaluaciones.csv"
SALIDA = RAIZ / "figuras" / "curva_busqueda.png"

AZUL = "#2a78d6"     # serie principal
GRIS = "#9a9893"     # evaluaciones individuales (contexto)
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#e4e3df"
BANDA = "#f1f0ec"

ETAPAS = [  # (desde, hasta, nombre) por número de evaluación
    (1, 36, "Fase 1\npasada 1"),
    (37, 47, "Pasada 2"),
    (48, 88, "Fase 2: exhaustivo sobre 48 configuraciones"),
    (89, 90, "Temp."),
]

with open(REGISTRO, encoding="utf-8") as f:
    filas = list(csv.DictReader(f))
n = [int(x["n"]) for x in filas]
aciertos = [int(x["aciertos"]) for x in filas]
mejor, tope = [], 0
for a in aciertos:
    tope = max(tope, a)
    mejor.append(tope)

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
fig.patch.set_facecolor("white")

for i, (desde, hasta, nombre) in enumerate(ETAPAS):
    if i % 2 == 0:
        ax.axvspan(desde - 0.5, hasta + 0.5, color=BANDA, zorder=0, lw=0)
    ax.text((desde + hasta) / 2, 19.6, nombre, ha="center", va="top", fontsize=9, color=TINTA_2)

ax.scatter(n, aciertos, s=14, color=GRIS, zorder=2, label="Precisión de cada configuración evaluada")
ax.step(n, mejor, where="post", color=AZUL, lw=2, zorder=3, label="Mejor resultado hasta esa evaluación")

# Marcadores y etiquetas solo donde mejora el mejor resultado
mejoras = [(k, m) for k, (m, previo) in enumerate(zip(mejor, [-1] + mejor[:-1])) if m > previo]
ax.scatter([n[k] for k, _ in mejoras], [m for _, m in mejoras], s=36, color=AZUL,
           edgecolor="white", linewidth=1.5, zorder=4)
for k, texto in [(0, "inicio 8 %"), (37, "fin de la fase 1: 14 %"), (84, "elegida: 17 %")]:
    ax.annotate(texto, (n[k], mejor[k]), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=9, color=TINTA)

ax.set_xlim(0, 91)
ax.set_ylim(0, 20)
ax.set_yticks(range(0, 21, 5), [f"{v} %" for v in range(0, 21, 5)])
ax.set_xlabel("Evaluación (configuraciones distintas, en orden)", color=TINTA)
ax.set_ylabel("Precisión en búsqueda (100 instancias)", color=TINTA)
ax.grid(axis="y", color=REJILLA, lw=1)
ax.set_axisbelow(True)
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
for lado in ("left", "bottom"):
    ax.spines[lado].set_color(TINTA_2)
ax.tick_params(colors=TINTA_2)
fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=9)

fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(SALIDA, facecolor="white")
print("figura:", SALIDA)
