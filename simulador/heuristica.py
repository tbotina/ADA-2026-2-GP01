"""
heuristica.py — búsqueda local por coordenadas, voraz por categoría (Alg. 2.3.2).

    from simulador.modelo import instancia_base
    from simulador.heuristica import busqueda_local, voraz

    r = busqueda_local(instancia_base(), inicial=(0, 0, 0, 0, 0))
    r.aciertos, r.config, r.pasadas, r.evaluaciones, r.convergio

    voraz(instancia_base())   # una sola pasada desde ninguna opción elegida (Teorema 2.3.1)

Recorre las categorías en orden; en cada una fija las demás, evalúa todas sus opciones y se
queda con la mejor. Una pasada cuesta $\\sum_j k_j \\le mK$ evaluaciones de $O(n\\rho)$, así que
$T(m, K, n, \\rho, I) = O(I \\cdot m \\cdot K \\cdot n \\cdot \\rho)$: polinomial, a diferencia del exacto.
"""

from collections import namedtuple

from simulador.modelo import aciertos

Resultado = namedtuple("Resultado", "aciertos config pasadas evaluaciones convergio")

# Una categoría sin opción elegida no cubre ninguna regla: es el punto de partida del voraz.
SIN_OPCION = None


def aciertos_parcial(instancia, config):
    """Como `aciertos`, pero las categorías en `SIN_OPCION` no aportan reglas."""
    cubiertas = 0
    for j, o in enumerate(config):
        if o is not SIN_OPCION:
            cubiertas |= instancia.opciones[j][o]
    faltan = ~cubiertas
    return sum(1 for requisito in instancia.requisitos if requisito & faltan == 0)


def busqueda_local(instancia, inicial=None, max_pasadas=10):
    """Búsqueda local por coordenadas desde `inicial`.

    Parameters
    ----------
    instancia:
        Una `Instancia` del modelo común.
    inicial:
        Configuración de partida. Con `None` se parte de ninguna opción elegida, y la primera
        pasada es la voraz del Teorema 2.3.1.
    max_pasadas:
        Tope $I$ de pasadas. El recorrido para antes si una pasada no cambia nada.

    Returns
    -------
    `Resultado(aciertos, config, pasadas, evaluaciones, convergio)`. `evaluaciones` cuenta las
    llamadas a la función objetivo, cada una de costo $O(n\\rho)$. En empate se conserva la
    opción actual; si no la hay, se toma la de menor índice. Así el resultado es determinista.
    """
    if max_pasadas < 1:
        raise ValueError("max_pasadas debe ser al menos 1")
    if inicial is None:
        actual = [SIN_OPCION] * instancia.m
    else:
        if len(inicial) != instancia.m:
            raise ValueError(f"la configuración tiene {len(inicial)} categorías; se esperan {instancia.m}")
        actual = list(inicial)

    evaluaciones = 0
    convergio = False
    pasadas = 0
    for pasadas in range(1, max_pasadas + 1):
        cambios = 0
        for j, opciones in enumerate(instancia.opciones):
            previa = actual[j]
            mejor_o, mejor_valor = previa, -1
            for o in range(len(opciones)):
                actual[j] = o
                valor = aciertos_parcial(instancia, actual)
                evaluaciones += 1
                # Solo se cambia de opción con una mejora estricta sobre la actual.
                if valor > mejor_valor or (valor == mejor_valor and o == previa):
                    mejor_o, mejor_valor = o, valor
            actual[j] = mejor_o
            cambios += mejor_o != previa
        if cambios == 0:
            convergio = True
            break

    config = tuple(actual)
    return Resultado(aciertos(instancia, config), config, pasadas, evaluaciones, convergio)


def voraz(instancia):
    """Una sola pasada voraz partiendo de ninguna opción elegida (Teorema 2.3.1)."""
    return busqueda_local(instancia, inicial=None, max_pasadas=1)
