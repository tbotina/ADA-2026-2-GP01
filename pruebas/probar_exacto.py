"""Pruebas del backtracking con poda (Alg. 2.3.1): solo biblioteca estándar.

La referencia es `fuerza_bruta`, que evalúa todas las configuraciones sin ninguna cota. Los casos
de la poda se calcularon a mano, nodo por nodo.

Uso, desde cualquier carpeta:

    python pruebas/probar_exacto.py
"""
import itertools
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulador.exacto import backtracking, cota_superior, nodos_peor_caso  # noqa: E402
from simulador.modelo import (  # noqa: E402
    Instancia,
    aciertos,
    configuraciones,
    desde_3sat,
    fuerza_bruta,
    generar,
    instancia_base,
)


def bits(*reglas):
    return sum(1 << r for r in reglas)


class CasosCalculadosAMano(unittest.TestCase):
    def test_poda_de_un_nodo_interno(self):
        """Reglas a=0, b=1. Recorrido: raíz, A1, (A1,B1)=3, (A1,B2)=1, A2 con UB=1: se corta."""
        instancia = Instancia(((bits(0), 0), (bits(1), 0)), (bits(0), bits(1), bits(0, 1)), 2)
        r = backtracking(instancia)
        self.assertEqual((r.aciertos, r.config), (3, (0, 0)))
        self.assertEqual((r.nodos, r.podas), (5, 1))

    def test_la_poda_tambien_corta_los_empates(self):
        """Tras (A1,B1)=1, A2 tiene UB=1: no *supera* a la mejor, así que se corta."""
        instancia = Instancia(((bits(0), bits(1)), (0, 0)), (bits(0), bits(1)), 2)
        r = backtracking(instancia)
        self.assertEqual((r.aciertos, r.config), (1, (0, 0)))
        self.assertEqual((r.nodos, r.podas), (5, 1))

    def test_una_categoria(self):
        instancia = Instancia(((bits(0), bits(0, 1), bits(1)),), (bits(0), bits(1)), 2)
        r = backtracking(instancia)
        self.assertEqual((r.aciertos, r.config), (2, (1,)))

    def test_sin_instancias_devuelve_la_primera_configuracion(self):
        r = backtracking(Instancia(((0, 0), (0,)), (), 1))
        self.assertEqual((r.aciertos, r.config), (0, (0, 0)))

    def test_regla_que_ninguna_opcion_cubre(self):
        r = backtracking(Instancia(((0, 0),), (bits(0),), 1))
        self.assertEqual((r.aciertos, r.config), (0, (0,)))

    def test_contraejemplo_del_enunciado(self):
        q = 7
        a1, a2, b1, b2 = 0, 1, 2, 3
        instancia = Instancia(((bits(a1), bits(a2)), (bits(b1), bits(b2))),
                              (bits(a1, b1),) + (bits(a2, b2),) * q, 4)
        r = backtracking(instancia)
        self.assertEqual((r.aciertos, r.config), (q, (1, 1)))


class CoincideConLaFuerzaBruta(unittest.TestCase):
    def comparar(self, instancia):
        r = backtracking(instancia)
        self.assertEqual((r.aciertos, r.config), fuerza_bruta(instancia))
        self.assertEqual(aciertos(instancia, r.config), r.aciertos)

    def test_instancia_base_con_varias_semillas(self):
        for semilla in range(30):
            self.comparar(instancia_base(semilla))

    def test_instancias_aleatorias_de_todas_las_formas(self):
        azar = random.Random(11)
        for _ in range(400):
            k = tuple(azar.randint(1, 5) for _ in range(azar.randint(1, 5)))
            rho = azar.randint(2, 25)
            minimo = azar.randint(1, min(3, rho))
            instancia = generar(k, azar.randint(1, 60), rho, azar.randrange(10**6),
                                prob_regla=azar.choice((0, 0.05, 0.2, 0.5, 0.9, 1)),
                                reglas_por_instancia=(minimo, azar.randint(minimo, min(minimo + 3, rho))))
            self.comparar(instancia)

    def test_sin_poda_da_el_mismo_optimo(self):
        for semilla in range(10):
            instancia = instancia_base(semilla)
            con, sin = backtracking(instancia), backtracking(instancia, podar=False)
            self.assertEqual((con.aciertos, con.config), (sin.aciertos, sin.config))

    def test_formulas_3sat(self):
        azar = random.Random(3)
        for _ in range(100):
            variables = azar.randint(3, 8)
            formula = [tuple(v * azar.choice((1, -1)) for v in azar.sample(range(1, variables + 1), 3))
                       for _ in range(azar.randint(1, 50))]
            self.comparar(desde_3sat(formula, variables))

    def test_formula_insatisfacible(self):
        formula = [tuple(s * v for s, v in zip(signos, (1, 2, 3)))
                   for signos in itertools.product((1, -1), repeat=3)]
        self.assertEqual(backtracking(desde_3sat(formula, 3)).aciertos, 7)


class LaCotaEsValida(unittest.TestCase):
    """Lo que exige el enunciado: ninguna completación de una rama resuelve más de UB instancias."""

    def test_ninguna_completacion_supera_la_cota(self):
        azar = random.Random(5)
        for _ in range(40):
            k = tuple(azar.randint(1, 4) for _ in range(azar.randint(1, 4)))
            instancia = generar(k, 40, 15, azar.randrange(10**6), prob_regla=azar.choice((0.1, 0.2, 0.4)))
            valores = {c: aciertos(instancia, c) for c in configuraciones(instancia)}
            for largo in range(instancia.m + 1):
                for prefijo in itertools.product(*(range(kj) for kj in k[:largo])):
                    cota = cota_superior(instancia, prefijo)
                    completaciones = [v for c, v in valores.items() if c[:largo] == prefijo]
                    self.assertGreaterEqual(cota, max(completaciones), prefijo)

    def test_en_una_hoja_la_cota_es_el_valor_exacto(self):
        instancia = instancia_base()
        for config in configuraciones(instancia):
            self.assertEqual(cota_superior(instancia, config), aciertos(instancia, config))

    def test_en_la_raiz_cuenta_lo_que_cubren_todas_las_opciones(self):
        # Nada exige la regla 2, y la regla 3 no la cubre ninguna opción.
        instancia = Instancia(((bits(0), bits(1)), (bits(2), 0)), (bits(0, 1), bits(3), bits(0, 3)), 4)
        self.assertEqual(cota_superior(instancia, ()), 1)


class CuentaDeNodos(unittest.TestCase):
    def test_arbol_completo_de_la_instancia_base(self):
        # 1 + 4 + 12 + 36 + 72 + 144
        self.assertEqual(nodos_peor_caso((4, 3, 3, 2, 2)), 269)

    def test_sin_poda_recorre_exactamente_el_arbol_completo(self):
        azar = random.Random(7)
        for _ in range(50):
            k = tuple(azar.randint(1, 4) for _ in range(azar.randint(1, 5)))
            r = backtracking(generar(k, 20, 10, azar.randrange(10**6)), podar=False)
            self.assertEqual((r.nodos, r.podas), (nodos_peor_caso(k), 0))

    def test_con_poda_nunca_recorre_mas(self):
        azar = random.Random(8)
        for _ in range(100):
            k = tuple(azar.randint(1, 4) for _ in range(azar.randint(1, 5)))
            r = backtracking(generar(k, 30, 12, azar.randrange(10**6)))
            self.assertLessEqual(r.nodos, nodos_peor_caso(k))

    def test_a_lo_sumo_el_doble_de_hojas_si_todo_kj_es_al_menos_2(self):
        azar = random.Random(9)
        for _ in range(200):
            k = tuple(azar.randint(2, 6) for _ in range(azar.randint(1, 8)))
            hojas = 1
            for kj in k:
                hojas *= kj
            self.assertLessEqual(nodos_peor_caso(k), 2 * hojas)

    def test_la_cota_de_2C_no_aplica_con_categorias_de_una_opcion(self):
        # El enunciado la condiciona a k_j >= 2: aquí hay 1 hoja y 4 nodos.
        self.assertEqual(nodos_peor_caso((1, 1, 1)), 4)

    def test_la_poda_sirve_en_la_instancia_base(self):
        for semilla in range(10):
            r = backtracking(instancia_base(semilla))
            self.assertLess(r.nodos, 269)
            self.assertGreater(r.podas, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
