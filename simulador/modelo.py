"""
modelo.py — el modelo común del enunciado (Definición 2.2.1), sin GPU ni dependencias.

    from simulador.modelo import instancia_base, aciertos, fuerza_bruta

    instancia = instancia_base()
    aciertos(instancia, (0, 0, 0, 0, 0))     # instancias resueltas por esa configuración
    fuerza_bruta(instancia)                  # (aciertos del óptimo, configuración)

Una configuración es una tupla con el índice de la opción elegida en cada categoría. Los
conjuntos de reglas son vectores de bits (enteros de Python): el bit `r` encendido significa
que la regla `r` está en el conjunto. Así, la unión es `|` y `A ⊆ B` es `A & ~B == 0`.
"""

import itertools
import math
import random
from collections import namedtuple

# Instancia base del enunciado: k = (4, 3, 3, 2, 2), |C| = 144, n = 150.
K_BASE = (4, 3, 3, 2, 2)
N_BASE = 150

# El enunciado deja libres estos parámetros. Se calibraron con 10 semillas: el óptimo fue
# único en todas (la Parte 2 debe identificar una sola mejor) y resolvió de 66 a 106 de las 150
# instancias, con una media de 52 entre las configuraciones. Ninguna configuración dio 0.
RHO_BASE = 30
PROB_REGLA_BASE = 0.2
REGLAS_POR_INSTANCIA_BASE = (2, 3)
SEMILLA_BASE = 2026


class Instancia(namedtuple("Instancia", "opciones requisitos rho")):
    """Una instancia del modelo común.

    Attributes
    ----------
    opciones:
        `opciones[j][o]` es el conjunto $S_o$ de la opción `o` de la categoría `j`.
    requisitos:
        `requisitos[i]` es el conjunto $R_i$ de reglas que exige la instancia de prueba `i`.
    rho:
        Tamaño del universo de reglas.
    """

    __slots__ = ()

    @property
    def m(self):
        """Número de categorías."""
        return len(self.opciones)

    @property
    def k(self):
        """Número de opciones de cada categoría."""
        return tuple(len(o) for o in self.opciones)

    @property
    def K(self):
        """Máximo de opciones por categoría."""
        return max(self.k)

    @property
    def n(self):
        """Número de instancias de prueba."""
        return len(self.requisitos)

    @property
    def tamano_espacio(self):
        """$|C|$: número de configuraciones."""
        return math.prod(self.k)


def reglas_de(instancia, config):
    """$S(c)$: unión de los conjuntos de las opciones elegidas."""
    if len(config) != instancia.m:
        raise ValueError(f"la configuración tiene {len(config)} categorías; se esperan {instancia.m}")
    cubiertas = 0
    for j, o in enumerate(config):
        if not 0 <= o < len(instancia.opciones[j]):
            raise ValueError(f"la opción {o} no existe en la categoría {j}")
        cubiertas |= instancia.opciones[j][o]
    return cubiertas


def resuelve(instancia, config, i):
    """$\\mathrm{res}(c, i)$: si la configuración cubre todas las reglas de la instancia `i`."""
    return instancia.requisitos[i] & ~reglas_de(instancia, config) == 0


def aciertos(instancia, config):
    """Número de instancias resueltas: la evaluación es todo-o-nada por instancia."""
    no_cubiertas = ~reglas_de(instancia, config)
    return sum(1 for requisito in instancia.requisitos if requisito & no_cubiertas == 0)


def precision(instancia, config):
    """$\\mathrm{Acc}(c)$: fracción de instancias resueltas."""
    return aciertos(instancia, config) / instancia.n


def configuraciones(instancia):
    """Todas las configuraciones, en orden lexicográfico."""
    return itertools.product(*(range(k) for k in instancia.k))


def fuerza_bruta(instancia):
    """Óptimo exacto: `(aciertos, config)`, evaluando las $|C|$ configuraciones.

    Es la referencia contra la que se comparan los demás algoritmos. En un empate devuelve
    la primera configuración en orden lexicográfico.
    """
    mejor = (-1, None)
    for config in configuraciones(instancia):
        valor = aciertos(instancia, config)
        if valor > mejor[0]:
            mejor = (valor, config)
    return mejor


def generar(k, n, rho, semilla, prob_regla=PROB_REGLA_BASE, reglas_por_instancia=REGLAS_POR_INSTANCIA_BASE):
    """Genera una instancia sintética; la misma semilla da siempre la misma instancia.

    Parameters
    ----------
    k:
        Número de opciones de cada categoría, por ejemplo `(4, 3, 3, 2, 2)`.
    n:
        Número de instancias de prueba.
    rho:
        Tamaño del universo de reglas.
    semilla:
        Semilla del generador. Es obligatoria para que todo experimento sea reproducible.
    prob_regla:
        Probabilidad de que una opción satisfaga cada regla, de forma independiente.
    reglas_por_instancia:
        `(mínimo, máximo)` de reglas que exige una instancia. Cada instancia toma un tamaño
        uniforme en ese rango y luego reglas distintas al azar.
    """
    minimo, maximo = reglas_por_instancia
    if not k or min(k) < 1:
        raise ValueError("debe haber al menos una categoría, y cada una con al menos una opción")
    if n < 1 or rho < 1:
        raise ValueError("n y rho deben ser positivos")
    if not 0 <= prob_regla <= 1:
        raise ValueError("prob_regla debe estar entre 0 y 1")
    if not 1 <= minimo <= maximo <= rho:
        raise ValueError("reglas_por_instancia debe cumplir 1 <= mínimo <= máximo <= rho")

    azar = random.Random(semilla)  # local: no toca el estado global de `random`
    opciones = tuple(
        tuple(sum(1 << r for r in range(rho) if azar.random() < prob_regla) for _ in range(kj))
        for kj in k
    )
    requisitos = tuple(
        sum(1 << r for r in azar.sample(range(rho), azar.randint(minimo, maximo)))
        for _ in range(n)
    )
    return Instancia(opciones, requisitos, rho)


def instancia_base(semilla=SEMILLA_BASE):
    """La instancia base del enunciado, con los parámetros libres fijados arriba."""
    return generar(K_BASE, N_BASE, RHO_BASE, semilla)


def desde_3sat(clausulas, variables):
    """Reducción del enunciado (§2.3.2): de una fórmula 3-CNF a una instancia del modelo.

    La fórmula es satisfacible si y solo si alguna configuración resuelve las `n` instancias.
    Cada variable es una categoría cuya opción 0 es *verdadero* y la opción 1 es *falso*.

    Parameters
    ----------
    clausulas:
        Lista de cláusulas. Cada una es una tupla de literales: `3` es $x_3$ y `-3` es $\\neg x_3$.
    variables:
        Número de variables $m$.
    """
    if variables < 1 or not clausulas:
        raise ValueError("la fórmula debe tener al menos una variable y una cláusula")
    verdadero, falso = [0] * variables, [0] * variables
    for i, clausula in enumerate(clausulas):
        for literal in clausula:
            if not 1 <= abs(literal) <= variables:
                raise ValueError(f"el literal {literal} no corresponde a ninguna variable")
            (verdadero if literal > 0 else falso)[abs(literal) - 1] |= 1 << i
    opciones = tuple(zip(verdadero, falso))
    requisitos = tuple(1 << i for i in range(len(clausulas)))
    return Instancia(opciones, requisitos, len(clausulas))


def asignacion_de(config):
    """La asignación de verdad que corresponde a una configuración de `desde_3sat`."""
    return tuple(o == 0 for o in config)
