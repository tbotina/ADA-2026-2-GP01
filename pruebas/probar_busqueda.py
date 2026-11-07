"""Pruebas de la búsqueda: no necesitan GPU, Colab ni el oráculo real.

Ejecutan el código de `notebooks/busqueda.ipynb` (secciones 3 a 10, tal como está en el
notebook, sin copiarlo) contra oráculos falsos cuyo resultado se conoce de antemano:

- `Reproduccion`: un oráculo que repite los aciertos de `registro_evaluaciones.csv`. La
  búsqueda debe recorrer las mismas 90 configuraciones en el mismo orden y elegir la misma.
- `ModeloSinInteraccion`: cada ranura tiene una opción limpia. Debe hallar el óptimo global.
- `ModeloConInteraccion`: dos opciones solo sirven juntas. La fase 1 se queda en un óptimo
  local y la fase 2 debe salir de él.
- `LimiteDeLaPoda`: documenta que la poda puede perder el óptimo; no hay garantía global.
- `ReglasDeLaBusqueda`: presupuesto, temperatura, validación y determinismo.

El óptimo de referencia se calcula por fuerza bruta sobre las 32 768 configuraciones. Estas
pruebas verifican la lógica del algoritmo; no dicen qué tan buena es la config en el oráculo real.

Uso, desde cualquier carpeta (requiere pandas, que el notebook usa para sus tablas):

    python pruebas/probar_busqueda.py
"""
import contextlib
import csv
import io
import itertools
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import pandas  # noqa: F401
except ImportError:
    sys.exit("Estas pruebas requieren pandas: pip install pandas")

RAIZ = Path(__file__).resolve().parent.parent
RANURAS = ["rol", "estrategia", "formato", "estilo", "cierre"]
TEMPERATURAS = [0.0, 0.3, 0.7]
CONFIGS = [dict(zip(RANURAS, c), temperatura=0.0) for c in itertools.product(range(8), repeat=5)]
INICIO = (1, 7, 1, 7, 6)  # la config inicial que fija el notebook con SEMILLA = 2026


class Resultado:
    def __init__(self, aciertos, trazas, n):
        self.precision, self.trazas, self.n = aciertos / n, trazas, n


class OraculoFalso:
    """Misma interfaz pública que el oráculo real; las subclases definen quién falla."""

    def __init__(self):
        self.llamadas = []
        self.validaciones = []

    def fallidas(self, config, instancias):
        raise NotImplementedError

    def evaluar(self, config, instancias, semilla=1):
        self.llamadas.append((dict(config), [x["id"] for x in instancias], semilla))
        trazas = [{"id": x["id"], "violo": violo, "salida": "..."}
                  for x, violo in self.fallidas(config, instancias)]
        return Resultado(len(instancias) - len(trazas), trazas, len(instancias))

    def validar(self, config, n=None):
        self.validaciones.append(dict(config))
        return Resultado(15, [{"id": i, "violo": "x:y", "salida": "..."} for i in range(72)], 87)


class OraculoModelo(OraculoFalso):
    """Una instancia falla si la config daña alguna de sus reglas.

    `danios[ranura][opcion]` son las reglas que daña esa opción. `bonos[temperatura]` son
    instancias fallidas que esa temperatura rescata, para probar la regla de temperatura.
    """

    def __init__(self, danios, reglas_por_instancia, bonos=None):
        super().__init__()
        self.danios = danios
        self.reglas = reglas_por_instancia
        self.bonos = bonos or {}

    def fallidas(self, config, instancias):
        danadas = set().union(*(self.danios[r][config[r]] for r in RANURAS))
        salida = [(x, sorted(self.reglas[x["id"]] & danadas)[0]) for x in instancias
                  if self.reglas[x["id"]] & danadas]
        return salida[self.bonos.get(config["temperatura"], 0):]

    def optimo(self):
        """(aciertos, config) exactos por fuerza bruta a temperatura 0.0."""
        indice = {r: i for i, r in enumerate(sorted(set().union(*self.reglas.values())))}
        mascara = lambda reglas: sum(1 << indice[r] for r in reglas if r in indice)
        grupos = {}
        for reglas in self.reglas.values():
            grupos[mascara(reglas)] = grupos.get(mascara(reglas), 0) + 1
        grupos = list(grupos.items())
        por_ranura = [[mascara(self.danios[r][o]) for o in range(8)] for r in RANURAS]
        mejor = (-1, None)
        for c in itertools.product(range(8), repeat=5):
            danadas = 0
            for j, o in enumerate(c):
                danadas |= por_ranura[j][o]
            aciertos = sum(k for m, k in grupos if not m & danadas)
            if aciertos > mejor[0]:
                mejor = (aciertos, c)
        return mejor


class OraculoReplay(OraculoFalso):
    """Repite los aciertos del registro; si le piden otra config, la prueba falla."""

    def __init__(self, filas):
        super().__init__()
        self.aciertos = {(tuple(int(f[r]) for r in RANURAS), float(f["temperatura"])): int(f["aciertos"])
                         for f in filas}

    def fallidas(self, config, instancias):
        clave = (tuple(config[r] for r in RANURAS), config["temperatura"])
        assert clave in self.aciertos, f"la búsqueda se salió de la corrida registrada: {clave}"
        return [(x, "familia:regla") for x in instancias[self.aciertos[clave]:]]


def celdas_de_busqueda():
    """Código del notebook desde los parámetros hasta la validación (sin la entrega)."""
    nb = json.loads((RAIZ / "notebooks" / "busqueda.ipynb").read_text(encoding="utf-8"))
    codigo = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    inicio = next(i for i, s in enumerate(codigo) if "INSTANCIAS = busqueda" in s)
    fin = next(i for i, s in enumerate(codigo) if "from ayudas import entrega" in s)
    return codigo[inicio:fin]


def correr_busqueda(oraculo, instancias=None):
    """Ejecuta el notebook contra `oraculo` y devuelve sus variables."""
    instancias = instancias or [{"id": i} for i in range(100)]
    ayudas = types.ModuleType("ayudas")
    ayudas.curva = lambda historial: None
    sys.modules["ayudas"] = ayudas
    ns = {"busqueda": instancias, "validacion": [{"id": 1000 + i} for i in range(87)],
          "CONFIGS": CONFIGS, "RANURAS": RANURAS, "TEMPERATURAS": TEMPERATURAS,
          "oraculo": oraculo, "display": lambda tabla: None}
    with tempfile.TemporaryDirectory() as carpeta, contextlib.redirect_stdout(io.StringIO()):
        for celda in celdas_de_busqueda():
            celda = celda.replace('"/content/drive/MyDrive/oraculo_adag1"', repr(carpeta))
            exec(compile(celda, "<notebook>", "exec"), ns)
    return ns


def tupla(config):
    return tuple(config[r] for r in RANURAS)


def aciertos_de(ns, config):
    return ns["aciertos"](ns["RESULTADOS"][ns["clave"](config)])


def modelo_sin_interaccion(semilla=7):
    """Cada ranura daña reglas de su propia familia, y una opción por ranura no daña nada."""
    azar = random.Random(semilla)
    limpia = {"rol": 6, "estrategia": 3, "formato": 4, "estilo": 1, "cierre": 7}
    familias = {r: [f"{r}:{k}" for k in range(4)] for r in RANURAS}
    danios = {r: {o: set() if o == limpia[r] else set(azar.sample(familias[r], azar.randint(1, 3)))
                  for o in range(8)} for r in RANURAS}
    reglas = {}
    for i in range(100):
        elegidas = azar.sample(RANURAS, azar.choice([2, 3]))
        reglas[i] = {azar.choice(familias[r]) for r in elegidas}
    return OraculoModelo(danios, reglas), limpia


def modelo_interaccion(rivales=False, bonos=None):
    """`rol 0` y `estrategia 5` solo resuelven el grupo conjunto si van juntas.

    Mientras no coincidan, `rol 3` y `estrategia 7` parecen mejores porque cada una resuelve
    unas pocas instancias propias. Con `rivales`, esas pocas son más y otra opción empata con
    ellas, de modo que la poda descarta `rol 0` y `estrategia 5`.
    """
    propias = 5 if rivales else 2
    buenas = {"rol": {0, 3, 4} if rivales else {0, 3}, "estrategia": {5, 7, 2} if rivales else {5, 7},
              "formato": {1}, "estilo": {7}, "cierre": {6}}
    danios = {r: {o: set() for o in range(8)} for r in RANURAS}
    for r in RANURAS:  # las opciones que no están en `buenas` dañan a su grupo base
        for o in set(range(8)) - buenas[r]:
            danios[r][o].add(f"base:{r}")
    for o in range(8):
        if o != 0:
            danios["rol"][o].add("junta:rol")
        if o != 5:
            danios["estrategia"][o].add("junta:estrategia")
        if o not in ({3, 4} if rivales else {3}):
            danios["rol"][o].add("propia:rol")
        if o not in ({7, 2} if rivales else {7}):
            danios["estrategia"][o].add("propia:estrategia")
        danios["formato"][o].add("imposible:x")
    grupos = [({f"base:{r}"}, 5) for r in RANURAS]
    grupos += [({"junta:rol", "junta:estrategia"}, 20), ({"propia:rol"}, propias), ({"propia:estrategia"}, propias)]
    reglas, i = {}, 0
    for conjunto, cuantas in grupos:
        for _ in range(cuantas):
            reglas[i] = set(conjunto)
            i += 1
    while i < 100:  # el resto no se resuelve con ninguna config
        reglas[i] = {"imposible:x"}
        i += 1
    return OraculoModelo(danios, reglas, bonos)


class Reproduccion(unittest.TestCase):
    """Resultado esperado real: la corrida guardada en notebooks/registro_evaluaciones.csv."""

    @classmethod
    def setUpClass(cls):
        with open(RAIZ / "notebooks" / "registro_evaluaciones.csv", encoding="utf-8") as f:
            cls.filas = list(csv.DictReader(f))
        cls.ns = correr_busqueda(OraculoReplay(cls.filas))

    def test_mismo_recorrido_en_el_mismo_orden(self):
        recorrido = [(tupla(f), f["temperatura"]) for f in self.ns["REGISTRO"]]
        registrado = [(tuple(int(f[r]) for r in RANURAS), float(f["temperatura"])) for f in self.filas]
        self.assertEqual(recorrido, registrado)
        self.assertEqual(len(recorrido), 90)

    def test_fin_de_la_fase_1(self):
        self.assertEqual(tupla(self.ns["FIN_FASE1"]), (3, 7, 3, 1, 3))

    def test_subespacio(self):
        self.assertEqual(self.ns["CANDIDATAS"], {"rol": [3, 0], "estrategia": [7, 5], "formato": [3, 0],
                                                 "estilo": [1, 2, 5], "cierre": [3, 6]})

    def test_config_elegida_es_la_entregada(self):
        entrega = json.loads((RAIZ / "entrega.json").read_text(encoding="utf-8"))
        self.assertEqual(self.ns["mejor"][1], entrega["config"])
        self.assertEqual(self.ns["mejor"][0], 0.17)


class ModeloSinInteraccion(unittest.TestCase):
    def test_halla_el_optimo_global(self):
        oraculo, limpia = modelo_sin_interaccion()
        aciertos, optimo = oraculo.optimo()
        self.assertEqual((aciertos, optimo), (100, tuple(limpia[r] for r in RANURAS)))
        ns = correr_busqueda(oraculo)
        self.assertEqual(tupla(ns["mejor"][1]), optimo)
        self.assertEqual(ns["mejor"][0], 1.0)


class ModeloConInteraccion(unittest.TestCase):
    """Óptimo local de las coordenadas (cf. Ejemplo 2.1 del enunciado)."""

    @classmethod
    def setUpClass(cls):
        cls.oraculo = modelo_interaccion()
        cls.ns = correr_busqueda(cls.oraculo)

    def test_fase_1_queda_en_un_optimo_local(self):
        self.assertEqual(tupla(self.ns["FIN_FASE1"]), (3, 7, 1, 7, 6))
        self.assertEqual(aciertos_de(self.ns, self.ns["FIN_FASE1"]), 29)

    def test_fase_2_sale_del_optimo_local(self):
        aciertos, optimo = self.oraculo.optimo()
        self.assertEqual((aciertos, optimo), (45, (0, 5, 1, 7, 6)))
        self.assertEqual(tupla(self.ns["mejor"][1]), optimo)

    def test_fase_2_es_exacta_en_el_subespacio(self):
        esperado = 0
        for combo in itertools.product(*(self.ns["CANDIDATAS"][r] for r in RANURAS)):
            config = dict(zip(RANURAS, combo), temperatura=0.0)
            esperado = max(esperado, 100 - len(self.oraculo.fallidas(config, self.ns["INSTANCIAS"])))
        self.assertEqual(round(self.ns["mejor"][0] * 100), esperado)


class LimiteDeLaPoda(unittest.TestCase):
    """La poda es empírica: aquí descarta las opciones del óptimo. Es la limitación del informe."""

    def test_puede_perder_el_optimo_global(self):
        oraculo = modelo_interaccion(rivales=True)
        aciertos, optimo = oraculo.optimo()
        self.assertEqual(optimo[:2], (0, 5))
        ns = correr_busqueda(oraculo)
        self.assertNotIn(0, ns["CANDIDATAS"]["rol"])
        self.assertNotIn(5, ns["CANDIDATAS"]["estrategia"])
        self.assertLess(round(ns["mejor"][0] * 100), aciertos)
        self.assertEqual(round(ns["mejor"][0] * 100), 35)  # mejor observada, no óptimo


class ReglasDeLaBusqueda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oraculo = modelo_interaccion()
        cls.ns = correr_busqueda(cls.oraculo)

    def test_cada_config_se_evalua_una_sola_vez(self):
        configs = [tuple(sorted(c.items())) for c, _, _ in self.oraculo.llamadas]
        self.assertEqual(len(configs), len(set(configs)))
        self.assertEqual(len(configs), len(self.ns["REGISTRO"]))

    def test_respeta_la_cota_de_evaluaciones(self):
        vivas = sum(len(v) - 1 for v in self.ns["VIVAS"].values())
        cota = (1 + 5 * 7) + vivas + self.ns["MAX_SUBESPACIO"] + (len(TEMPERATURAS) - 1)
        self.assertLessEqual(len(self.oraculo.llamadas), cota)

    def test_subespacio_no_supera_el_tope_y_contiene_la_fase_1(self):
        candidatas = self.ns["CANDIDATAS"]
        total = 1
        for r in RANURAS:
            total *= len(candidatas[r])
            self.assertIn(self.ns["FIN_FASE1"][r], candidatas[r])
        self.assertLessEqual(total, self.ns["MAX_SUBESPACIO"])

    def test_busca_siempre_sobre_las_mismas_100_instancias_y_semilla(self):
        for _, ids, semilla in self.oraculo.llamadas:
            self.assertEqual(ids, list(range(100)))
            self.assertEqual(semilla, 1)

    def test_valida_una_sola_vez_y_sobre_la_elegida(self):
        self.assertEqual(self.oraculo.validaciones, [self.ns["mejor"][1]])

    def test_el_resultado_nunca_empeora(self):
        inicial = aciertos_de(self.ns, dict(zip(RANURAS, INICIO), temperatura=0.0))
        fase_1 = aciertos_de(self.ns, self.ns["FIN_FASE1"])
        self.assertLessEqual(inicial, fase_1)
        self.assertLessEqual(fase_1, round(self.ns["mejor"][0] * 100))

    def test_es_determinista(self):
        otra = correr_busqueda(modelo_interaccion())
        quitar_tiempo = lambda filas: [{k: v for k, v in f.items() if k != "segundos"} for f in filas]
        self.assertEqual(quitar_tiempo(otra["REGISTRO"]), quitar_tiempo(self.ns["REGISTRO"]))

    def test_temperatura_no_se_adopta_si_no_supera_la_tolerancia(self):
        ns = correr_busqueda(modelo_interaccion(bonos={0.3: 2, 0.7: 1}))
        self.assertEqual(ns["mejor"][1]["temperatura"], 0.0)

    def test_temperatura_se_adopta_si_supera_la_tolerancia(self):
        ns = correr_busqueda(modelo_interaccion(bonos={0.3: 1, 0.7: 3}))
        self.assertEqual(ns["mejor"][1]["temperatura"], 0.7)
        self.assertEqual(round(ns["mejor"][0] * 100), 48)


if __name__ == "__main__":
    unittest.main(verbosity=2)
