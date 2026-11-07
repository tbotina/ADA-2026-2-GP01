"""Pruebas del simulador del modelo común (Definición 2.2.1): solo biblioteca estándar.

Los resultados esperados salen de casos calculados a mano (el ejemplo del informe, el
contraejemplo 2.1 del enunciado, una fórmula insatisfacible) y de comprobaciones independientes:
la precisión se recalcula con conjuntos de Python y la satisfacibilidad, evaluando la fórmula.

Uso, desde cualquier carpeta:

    python pruebas/probar_simulador.py
"""
import itertools
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulador.modelo import (  # noqa: E402
    Instancia,
    aciertos,
    asignacion_de,
    configuraciones,
    desde_3sat,
    fuerza_bruta,
    generar,
    instancia_base,
    precision,
    reglas_de,
    resuelve,
)


def bits(*reglas):
    return sum(1 << r for r in reglas)


def conjunto(vector):
    return {r for r in range(vector.bit_length()) if vector >> r & 1}


def satisface(clausula, asignacion):
    return any(asignacion[abs(lit) - 1] == (lit > 0) for lit in clausula)


class ModeloCalculadoAMano(unittest.TestCase):
    """Dos categorías y cuatro reglas: a=0, b=1, c=2, d=3."""

    def setUp(self):
        self.instancia = Instancia(
            opciones=((bits(0), bits(1)), (bits(2), bits(3), bits(0, 2))),
            requisitos=(bits(0), bits(0, 2), bits(1, 3), bits(0, 1)),
            rho=4,
        )

    def test_tamanos(self):
        i = self.instancia
        self.assertEqual((i.m, i.k, i.K, i.n, i.tamano_espacio), (2, (2, 3), 3, 4, 6))

    def test_una_configuracion_cubre_la_union_de_sus_opciones(self):
        self.assertEqual(reglas_de(self.instancia, (1, 1)), bits(1, 3))
        self.assertEqual(reglas_de(self.instancia, (1, 2)), bits(0, 1, 2))

    def test_todo_o_nada_exige_todas_las_reglas(self):
        self.assertTrue(resuelve(self.instancia, (0, 0), 1))  # {a, c} con {a} y {c}
        self.assertFalse(resuelve(self.instancia, (0, 1), 1))  # falta c
        self.assertFalse(resuelve(self.instancia, (0, 0), 3))  # {a, b}: tiene a, pero no b

    def test_aciertos_de_cada_configuracion(self):
        esperado = {(0, 0): 2, (0, 1): 1, (0, 2): 2, (1, 0): 0, (1, 1): 1, (1, 2): 3}
        for config, valor in esperado.items():
            self.assertEqual(aciertos(self.instancia, config), valor, config)
        self.assertEqual(precision(self.instancia, (1, 2)), 0.75)

    def test_fuerza_bruta_halla_el_optimo(self):
        self.assertEqual(fuerza_bruta(self.instancia), (3, (1, 2)))

    def test_rechaza_configuraciones_invalidas(self):
        for config in [(0,), (0, 0, 0), (2, 0), (0, 3), (-1, 0)]:
            with self.assertRaises(ValueError, msg=config):
                aciertos(self.instancia, config)


class ContraejemploDelEnunciado(unittest.TestCase):
    """Ejemplo 2.1: A1, A2, B1, B2 con una instancia {a1, b1} y q instancias {a2, b2}."""

    def test_valores(self):
        q = 7
        a1, a2, b1, b2 = 0, 1, 2, 3
        instancia = Instancia(((bits(a1), bits(a2)), (bits(b1), bits(b2))),
                              (bits(a1, b1),) + (bits(a2, b2),) * q, 4)
        self.assertEqual(aciertos(instancia, (0, 0)), 1)
        self.assertEqual(aciertos(instancia, (1, 0)), 0)
        self.assertEqual(aciertos(instancia, (0, 1)), 0)
        self.assertEqual(fuerza_bruta(instancia), (q, (1, 1)))


class ReduccionDesde3SAT(unittest.TestCase):
    # El ejemplo pequeño del informe (§5.8)
    FORMULA = [(1, -2, 3), (-1, 2, 3), (-1, -2, -3)]

    def test_opciones_del_ejemplo_del_informe(self):
        instancia = desde_3sat(self.FORMULA, 3)
        r1, r2, r3 = 0, 1, 2
        self.assertEqual(instancia.opciones, ((bits(r1), bits(r2, r3)), (bits(r2), bits(r1, r3)), (bits(r1, r2), bits(r3))))
        self.assertEqual(instancia.requisitos, (bits(r1), bits(r2), bits(r3)))
        self.assertEqual(instancia.k, (2, 2, 2))

    def test_configuracion_del_ejemplo_del_informe(self):
        instancia = desde_3sat(self.FORMULA, 3)
        config = (1, 1, 0)  # (F1, F2, V3)
        self.assertEqual(aciertos(instancia, config), 3)
        self.assertEqual(asignacion_de(config), (False, False, True))

    def test_formula_insatisfacible(self):
        # Las 8 cláusulas sobre 3 variables: cada asignación falsifica exactamente una.
        formula = [tuple(s * v for s, v in zip(signos, (1, 2, 3)))
                   for signos in itertools.product((1, -1), repeat=3)]
        instancia = desde_3sat(formula, 3)
        for config in configuraciones(instancia):
            self.assertEqual(aciertos(instancia, config), 7)
        self.assertEqual(fuerza_bruta(instancia)[0], 7)

    def test_equivalencia_en_formulas_aleatorias(self):
        """Proposición 2.3.1 y su lema, contra una evaluación directa de la fórmula."""
        azar = random.Random(1)
        satisfacibles = insatisfacibles = 0
        for _ in range(150):
            variables = azar.randint(3, 6)
            n = azar.randint(1, 40)
            formula = [tuple(v * azar.choice((1, -1)) for v in azar.sample(range(1, variables + 1), 3))
                       for _ in range(n)]
            instancia = desde_3sat(formula, variables)
            for config in configuraciones(instancia):
                asignacion = asignacion_de(config)
                for i, clausula in enumerate(formula):
                    self.assertEqual(resuelve(instancia, config, i), satisface(clausula, asignacion))
            es_satisfacible = any(all(satisface(c, a) for c in formula)
                                  for a in itertools.product((True, False), repeat=variables))
            self.assertEqual(fuerza_bruta(instancia)[0] == n, es_satisfacible)
            satisfacibles += es_satisfacible
            insatisfacibles += not es_satisfacible
        # La prueba no sirve si solo aparece uno de los dos casos.
        self.assertGreater(satisfacibles, 10)
        self.assertGreater(insatisfacibles, 10)

    def test_rechaza_literales_invalidos(self):
        for formula, variables in [([(1, 2, 4)], 3), ([(0, 1, 2)], 3), ([], 3), ([(1,)], 0)]:
            with self.assertRaises(ValueError, msg=formula):
                desde_3sat(formula, variables)


class Generador(unittest.TestCase):
    def test_instancia_base_del_enunciado(self):
        instancia = instancia_base()
        self.assertEqual((instancia.k, instancia.tamano_espacio, instancia.n), ((4, 3, 3, 2, 2), 144, 150))

    def test_misma_semilla_misma_instancia(self):
        self.assertEqual(instancia_base(5), instancia_base(5))
        self.assertNotEqual(instancia_base(5), instancia_base(6))

    def test_no_toca_el_azar_global(self):
        random.seed(0)
        esperado = random.random()
        random.seed(0)
        instancia_base(5)
        self.assertEqual(random.random(), esperado)

    def test_respeta_los_parametros(self):
        instancia = generar((3, 2), 200, 12, semilla=3, prob_regla=0.5, reglas_por_instancia=(2, 4))
        self.assertEqual((instancia.k, instancia.n, instancia.rho), ((3, 2), 200, 12))
        tamanos = {len(conjunto(r)) for r in instancia.requisitos}
        self.assertEqual(tamanos, {2, 3, 4})
        universo = (1 << 12) - 1
        for vector in list(instancia.requisitos) + [o for c in instancia.opciones for o in c]:
            self.assertEqual(vector & ~universo, 0)

    def test_probabilidades_extremas(self):
        vacia = generar((2, 2), 10, 5, semilla=1, prob_regla=0)
        llena = generar((2, 2), 10, 5, semilla=1, prob_regla=1)
        self.assertTrue(all(aciertos(llena, c) == 10 for c in configuraciones(llena)))
        # Todas empatan: la fuerza bruta debe devolver la primera en orden lexicográfico.
        self.assertEqual(fuerza_bruta(vacia), (0, (0, 0)))
        self.assertEqual(fuerza_bruta(llena), (10, (0, 0)))

    def test_rechaza_parametros_invalidos(self):
        casos = [dict(k=(), n=1, rho=1), dict(k=(2, 0), n=1, rho=1), dict(k=(2,), n=0, rho=1),
                 dict(k=(2,), n=1, rho=0), dict(k=(2,), n=1, rho=5, prob_regla=1.5),
                 dict(k=(2,), n=1, rho=5, reglas_por_instancia=(0, 2)),
                 dict(k=(2,), n=1, rho=5, reglas_por_instancia=(3, 2)),
                 dict(k=(2,), n=1, rho=5, reglas_por_instancia=(1, 6))]
        for caso in casos:
            with self.assertRaises(ValueError, msg=caso):
                generar(semilla=1, **caso)


class ContraUnaImplementacionIndependiente(unittest.TestCase):
    def test_vectores_de_bits_equivalen_a_conjuntos(self):
        """Recalcula la Definición 2.2.1 con conjuntos de Python, como la escribe el enunciado."""
        for semilla in range(5):
            instancia = instancia_base(semilla)
            opciones = [[conjunto(o) for o in categoria] for categoria in instancia.opciones]
            requisitos = [conjunto(r) for r in instancia.requisitos]
            valores = {}
            for config in configuraciones(instancia):
                cubiertas = set().union(*(opciones[j][o] for j, o in enumerate(config)))
                valores[config] = sum(r <= cubiertas for r in requisitos)
                self.assertEqual(aciertos(instancia, config), valores[config])
            self.assertEqual(len(valores), 144)
            self.assertEqual(fuerza_bruta(instancia)[0], max(valores.values()))

    def test_la_instancia_base_no_es_trivial(self):
        """Por la calibración: óptimo único, sin resolverlo todo, y no todas empatan."""
        instancia = instancia_base()
        valores = [aciertos(instancia, c) for c in configuraciones(instancia)]
        mejor = max(valores)
        self.assertEqual(valores.count(mejor), 1)
        self.assertLess(mejor, instancia.n)
        self.assertGreater(len(set(valores)), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
