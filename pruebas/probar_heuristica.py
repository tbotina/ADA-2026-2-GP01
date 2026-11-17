"""Pruebas de la búsqueda local por coordenadas (Alg. 2.3.2): solo biblioteca estándar.

La referencia es el backtracking (`simulador/exacto.py`), ya probado contra la fuerza bruta. Los
casos de empate, convergencia y número de evaluaciones se calcularon a mano.

Uso, desde cualquier carpeta:

    python pruebas/probar_heuristica.py
"""
import itertools
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulador.exacto import backtracking  # noqa: E402
from simulador.heuristica import SIN_OPCION, aciertos_parcial, busqueda_local, voraz  # noqa: E402
from simulador.instancias import contraejemplo  # noqa: E402
from simulador.modelo import Instancia, aciertos, desde_3sat, generar, instancia_base  # noqa: E402


def bits(*reglas):
    return sum(1 << r for r in reglas)


def formula_al_azar(azar, variables, clausulas):
    return [tuple(v * azar.choice((1, -1)) for v in azar.sample(range(1, variables + 1), 3))
            for _ in range(clausulas)]


class CasosCalculadosAMano(unittest.TestCase):
    def test_una_pasada_desde_una_configuracion(self):
        """Reglas a=0, b=1, c=2. Desde (1, 0): la categoría 1 pasa a 0 (2 aciertos) y la 2 a 1 (3)."""
        instancia = Instancia(((bits(0), 0), (bits(1), bits(1, 2))), (bits(0), bits(0, 1), bits(2)), 3)
        r = busqueda_local(instancia, inicial=(1, 0))
        self.assertEqual((r.aciertos, r.config), (3, (0, 1)))
        self.assertEqual((r.pasadas, r.convergio), (2, True))  # la segunda pasada no cambia nada
        self.assertEqual(r.evaluaciones, 2 * (2 + 2))

    def test_en_empate_conserva_la_opcion_actual(self):
        instancia = Instancia(((bits(0), bits(0)),), (bits(0),), 1)
        self.assertEqual(busqueda_local(instancia, inicial=(1,)).config, (1,))
        self.assertEqual(busqueda_local(instancia, inicial=(0,)).config, (0,))

    def test_sin_opcion_inicial_el_empate_toma_el_menor_indice(self):
        instancia = Instancia(((0, 0, 0),), (bits(0),), 1)
        self.assertEqual(voraz(instancia).config, (0,))

    def test_aciertos_parcial_ignora_las_categorias_vacias(self):
        instancia = Instancia(((bits(0),), (bits(1),)), (bits(0), bits(1), bits(0, 1)), 2)
        self.assertEqual(aciertos_parcial(instancia, [SIN_OPCION, SIN_OPCION]), 0)
        self.assertEqual(aciertos_parcial(instancia, [0, SIN_OPCION]), 1)
        self.assertEqual(aciertos_parcial(instancia, [0, 0]), 3)

    def test_tope_de_pasadas(self):
        """Cada pasada mejora una sola categoría y el tope corta antes de converger."""
        # Con (0,0,0) nada se resuelve. Ir a 1 en la categoría j+1 solo sirve si la j ya está en 1.
        instancia = Instancia(((0, bits(0)), (0, bits(1)), (0, bits(2))),
                              (bits(0),) * 3 + (bits(0, 1),) * 2 + (bits(0, 1, 2),), 3)
        r = busqueda_local(instancia, inicial=(0, 0, 0), max_pasadas=1)
        self.assertEqual((r.config, r.pasadas, r.convergio), ((1, 1, 1), 1, False))
        r = busqueda_local(instancia, inicial=(0, 0, 0), max_pasadas=10)
        self.assertEqual((r.config, r.pasadas, r.convergio), ((1, 1, 1), 2, True))

    def test_rechaza_parametros_invalidos(self):
        instancia = instancia_base()
        with self.assertRaises(ValueError):
            busqueda_local(instancia, inicial=(0, 0))
        with self.assertRaises(ValueError):
            busqueda_local(instancia, max_pasadas=0)


class ContraejemploDelEnunciado(unittest.TestCase):
    """Ejemplo 2.1: atrapada con valor 1 mientras el óptimo vale q."""

    def test_ejemplo_del_enunciado(self):
        for q in (1, 2, 10, 1000):
            instancia = contraejemplo(q)
            self.assertEqual(aciertos(instancia, (0, 0)), 1)
            self.assertEqual(aciertos(instancia, (1, 0)), 0)
            self.assertEqual(aciertos(instancia, (0, 1)), 0)
            self.assertEqual(aciertos(instancia, (1, 1)), q)
            r = busqueda_local(instancia, inicial=(0, 0))
            self.assertEqual((r.aciertos, r.config, r.convergio), (1, (0, 0), True))
            self.assertEqual(backtracking(instancia).aciertos, q)

    def test_generalizado_a_m_categorias(self):
        for m in (2, 3, 5, 8):
            instancia = contraejemplo(q=50, m=m)
            self.assertEqual(instancia.k, (2,) * m)
            inicial = (0,) * m
            self.assertEqual(aciertos(instancia, inicial), 1)
            for j in range(m):  # ningún vecino a una coordenada mejora
                vecino = inicial[:j] + (1,) + inicial[j + 1:]
                self.assertEqual(aciertos(instancia, vecino), 0)
            r = busqueda_local(instancia, inicial=inicial)
            self.assertEqual((r.aciertos, r.config), (1, inicial))
            self.assertEqual(backtracking(instancia).aciertos, 50)

    def test_la_razon_se_hace_arbitrariamente_mala(self):
        for q in (10, 100, 1000):
            r = busqueda_local(contraejemplo(q), inicial=(0, 0))
            self.assertEqual(r.aciertos / backtracking(contraejemplo(q)).aciertos, 1 / q)

    def test_la_trampa_depende_del_punto_de_partida(self):
        """Desde (0, 1): la categoría 0 pasa a 1 y se llega al óptimo (1, 1). Desde (1, 0): la
        categoría 0 se barre primero, (0, 0) vale 1 y gana, y se cae en la trampa."""
        r = busqueda_local(contraejemplo(10), inicial=(0, 1))
        self.assertEqual((r.aciertos, r.config), (10, (1, 1)))
        r = busqueda_local(contraejemplo(10), inicial=(1, 0))
        self.assertEqual((r.aciertos, r.config), (1, (0, 0)))

    def test_rechaza_parametros_invalidos(self):
        for q, m in [(0, 2), (1, 1)]:
            with self.assertRaises(ValueError):
                contraejemplo(q, m)


class GarantiaConUnaReglaPorInstancia(unittest.TestCase):
    """Teorema 2.3.1: con |R_i| = 1, el voraz desde ninguna opción da al menos la mitad del óptimo."""

    def test_en_reducciones_desde_3sat(self):
        azar = random.Random(21)
        peor = 1.0
        for _ in range(300):
            variables = azar.randint(3, 8)
            instancia = desde_3sat(formula_al_azar(azar, variables, azar.randint(1, 60)), variables)
            optimo = backtracking(instancia).aciertos
            valor = voraz(instancia).aciertos
            self.assertGreaterEqual(2 * valor, optimo)
            peor = min(peor, valor / optimo)
        self.assertLess(peor, 1.0)  # el voraz no siempre es óptimo: la cota no es trivial

    def test_en_instancias_aleatorias_con_una_regla(self):
        azar = random.Random(22)
        for _ in range(300):
            k = tuple(azar.randint(1, 5) for _ in range(azar.randint(1, 5)))
            instancia = generar(k, azar.randint(1, 60), azar.randint(2, 20), azar.randrange(10**6),
                                prob_regla=azar.choice((0.1, 0.3, 0.5)), reglas_por_instancia=(1, 1))
            self.assertGreaterEqual(2 * voraz(instancia).aciertos, backtracking(instancia).aciertos)

    def test_el_voraz_hace_una_sola_pasada(self):
        instancia = instancia_base()
        r = voraz(instancia)
        self.assertEqual((r.pasadas, r.evaluaciones), (1, sum(instancia.k)))

    def test_la_garantia_falla_con_varias_reglas(self):
        """Con |R_i| = 2 el voraz desde vacío también cae en la trampa del contraejemplo."""
        r = voraz(contraejemplo(10))
        self.assertEqual(r.aciertos, 1)
        self.assertLess(2 * r.aciertos, 10)


class PropiedadesGenerales(unittest.TestCase):
    def instancias(self, semilla, cuantas):
        azar = random.Random(semilla)
        for _ in range(cuantas):
            k = tuple(azar.randint(1, 5) for _ in range(azar.randint(1, 5)))
            yield generar(k, azar.randint(1, 60), azar.randint(3, 20), azar.randrange(10**6),
                          prob_regla=azar.choice((0.05, 0.2, 0.5)))

    def test_nunca_supera_al_exacto_y_el_valor_coincide_con_la_config(self):
        for instancia in self.instancias(31, 300):
            r = busqueda_local(instancia)
            self.assertEqual(aciertos(instancia, r.config), r.aciertos)
            self.assertLessEqual(r.aciertos, backtracking(instancia).aciertos)

    def test_cada_pasada_no_empeora(self):
        azar = random.Random(32)
        for instancia in self.instancias(33, 200):
            inicial = tuple(azar.randrange(kj) for kj in instancia.k)
            valores = [aciertos(instancia, inicial)]
            for pasadas in range(1, 5):
                valores.append(busqueda_local(instancia, inicial, max_pasadas=pasadas).aciertos)
            self.assertEqual(valores, sorted(valores))

    def test_al_converger_ningun_vecino_mejora(self):
        for instancia in self.instancias(34, 200):
            r = busqueda_local(instancia)
            self.assertTrue(r.convergio)
            for j, kj in enumerate(instancia.k):
                for o in range(kj):
                    vecino = r.config[:j] + (o,) + r.config[j + 1:]
                    self.assertLessEqual(aciertos(instancia, vecino), r.aciertos)

    def test_costo_de_evaluaciones(self):
        for instancia in self.instancias(35, 200):
            r = busqueda_local(instancia, max_pasadas=4)
            self.assertEqual(r.evaluaciones, r.pasadas * sum(instancia.k))
            self.assertLessEqual(r.evaluaciones, 4 * instancia.m * instancia.K)

    def test_es_determinista(self):
        instancia = instancia_base()
        self.assertEqual(busqueda_local(instancia), busqueda_local(instancia))
        self.assertEqual(busqueda_local(instancia, (3, 2, 2, 1, 1)), busqueda_local(instancia, (3, 2, 2, 1, 1)))

    def test_en_la_instancia_base(self):
        """Desde las 144 configuraciones, con a lo sumo 10 pasadas: nunca peor que el inicio."""
        instancia = instancia_base()
        optimo = backtracking(instancia).aciertos
        alcanzan = 0
        for inicial in itertools.product(*(range(kj) for kj in instancia.k)):
            r = busqueda_local(instancia, inicial)
            self.assertGreaterEqual(r.aciertos, aciertos(instancia, inicial))
            self.assertTrue(r.convergio)
            alcanzan += r.aciertos == optimo
        self.assertGreater(alcanzan, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
