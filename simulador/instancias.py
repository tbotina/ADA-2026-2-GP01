"""
instancias.py — instancias con estructura conocida, para pruebas y experimentos.

    from simulador.instancias import contraejemplo

    contraejemplo(q=10)        # el Ejemplo 2.1 del enunciado: la búsqueda local vale 1, el óptimo q
    contraejemplo(q=10, m=4)   # generalizado a m categorías
"""

from simulador.modelo import Instancia


def contraejemplo(q, m=2):
    """El contraejemplo para la búsqueda local (Ejemplo 2.1), generalizado a `m` categorías.

    Cada categoría `j` tiene dos opciones: la 0 satisface la regla $a_j$ y la 1 la regla $b_j$.
    Una instancia exige $\\{a_1, \\dots, a_m\\}$ y otras `q` exigen $\\{b_1, \\dots, b_m\\}$.

    Partiendo de la configuración con todas las opciones en 0 se resuelve una instancia, y
    cambiar una sola coordenada no resuelve ninguna, así que la búsqueda local se detiene con
    valor 1. El óptimo, con todas en 1, vale `q`. La razón $1/q$ se hace tan mala como se quiera.
    Con `m = 2` es exactamente el ejemplo del enunciado.
    """
    if q < 1 or m < 2:
        raise ValueError("se necesitan q >= 1 instancias y m >= 2 categorías")
    a = [1 << (2 * j) for j in range(m)]
    b = [1 << (2 * j + 1) for j in range(m)]
    opciones = tuple((a[j], b[j]) for j in range(m))
    requisitos = (sum(a),) + (sum(b),) * q
    return Instancia(opciones, requisitos, 2 * m)
