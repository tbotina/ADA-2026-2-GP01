"""
exacto.py — algoritmo exacto de la Parte 1: backtracking con poda por cota superior (Alg. 2.3.1).

    from simulador.modelo import instancia_base
    from simulador.exacto import backtracking

    r = backtracking(instancia_base())
    r.aciertos, r.config      # el óptimo
    r.nodos, r.podas          # cuánto recorrió, para compararlo con `nodos_peor_caso`
"""

from collections import namedtuple

Resultado = namedtuple("Resultado", "aciertos config nodos podas")


def nodos_peor_caso(k):
    """Nodos del árbol completo: la raíz más los nodos de cada nivel, $\\sum_j \\prod_{l \\le j} k_l$."""
    total = nivel = 1
    for kj in k:
        nivel *= kj
        total += nivel
    return total


def uniones_restantes(instancia):
    """`resto[j]` es $U_{>j}$: la unión de todas las opciones de las categorías desde `j`."""
    resto = [0] * (instancia.m + 1)
    for j in range(instancia.m - 1, -1, -1):
        resto[j] = resto[j + 1]
        for opcion in instancia.opciones[j]:
            resto[j] |= opcion
    return resto


def contar_cubiertas(requisitos, reglas):
    """Cuántos requisitos están contenidos en el conjunto `reglas`."""
    faltan = ~reglas
    return sum(1 for requisito in requisitos if requisito & faltan == 0)


def cota_superior(instancia, prefijo):
    """UB: instancias que podría resolver alguna configuración que empiece por `prefijo`."""
    cubiertas = 0
    for j, o in enumerate(prefijo):
        cubiertas |= instancia.opciones[j][o]
    return contar_cubiertas(instancia.requisitos, cubiertas | uniones_restantes(instancia)[len(prefijo)])


def backtracking(instancia, podar=True):
    """Óptimo exacto de la Parte 1, en el mismo orden lexicográfico que `fuerza_bruta`.

    Asigna una opción por categoría, en orden. En cada nodo calcula la cota UB y corta la rama
    si no supera a la mejor configuración hallada: ninguna completación podría mejorarla.

    Parameters
    ----------
    instancia:
        Una `Instancia` del modelo común.
    podar:
        Con `False` recorre el árbol completo, para medir el efecto de la poda.

    Returns
    -------
    `Resultado(aciertos, config, nodos, podas)`. `nodos` cuenta todos los nodos visitados y
    `podas`, los nodos internos en los que se cortó la rama.
    """
    m, requisitos, opciones = instancia.m, instancia.requisitos, instancia.opciones
    resto = uniones_restantes(instancia)
    mejor_valor, mejor_config = -1, None
    nodos = podas = 0
    prefijo = [0] * m

    def visitar(j, cubiertas):
        nonlocal mejor_valor, mejor_config, nodos, podas
        nodos += 1
        cota = contar_cubiertas(requisitos, cubiertas | resto[j])
        if j == m:  # en una hoja, $U_{>m}$ es vacío y la cota es el valor exacto
            if cota > mejor_valor:
                mejor_valor, mejor_config = cota, tuple(prefijo)
            return
        if podar and cota <= mejor_valor:
            podas += 1
            return
        for o, reglas in enumerate(opciones[j]):
            prefijo[j] = o
            visitar(j + 1, cubiertas | reglas)

    visitar(0, 0)
    return Resultado(mejor_valor, mejor_config, nodos, podas)
