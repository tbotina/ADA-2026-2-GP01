"""Pruebas locales de la entrega: no necesitan GPU, Colab ni dependencias.

Comprueban, sin consultar el oráculo, que

- `entrega.json` es válido: campos, tipos e índices que acepta la interfaz;
- el validador rechaza configuraciones que violan las restricciones del problema
  (casos con resultado esperado);
- la config de `entrega.json` es la misma que elige `notebooks/busqueda.ipynb`, la
  de más aciertos en `notebooks/registro_evaluaciones.csv` y la que documenta
  `notebooks/evaluacion.ipynb`.

Uso, desde cualquier carpeta:

    python pruebas/probar_entrega.py
"""
import ast
import csv
import json
import re
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# Copia de oraculo.RANURAS y oraculo.TEMPERATURAS (API pública): importar oraculo
# exige torch y GPU, y esta prueba debe correr en cualquier máquina.
RANURAS = ["rol", "estrategia", "formato", "estilo", "cierre"]
TEMPERATURAS = [0.0, 0.3, 0.7]
OPCIONES = range(8)

# Resultado esperado: la config elegida en la corrida del 30–31 oct 2026.
ESPERADA = {"rol": 0, "estrategia": 5, "formato": 0, "estilo": 2, "cierre": 3, "temperatura": 0.0}


def verificar_entrega(d):
    """Lista de problemas del JSON de entrega; vacía si es válido.

    Equivale a la comprobación de la sección 2 de `notebooks/evaluacion.ipynb`.
    """
    problemas = []
    if set(d) != {"grupo", "config", "semana"}:
        problemas.append(f"campos {sorted(d)}; se esperan grupo, config y semana")
    if not isinstance(d.get("grupo"), str) or not d.get("grupo"):
        problemas.append("grupo debe ser texto no vacío")
    if type(d.get("semana")) is not int:
        problemas.append("semana debe ser entero")
    config = d.get("config", {})
    if set(config) != set(RANURAS) | {"temperatura"}:
        problemas.append(f"config trae {sorted(config)}; se esperan {RANURAS} y temperatura")
    for ranura in RANURAS:
        if type(config.get(ranura)) is not int or config[ranura] not in OPCIONES:
            problemas.append(f"{ranura} debe ser un entero entre 0 y 7")
    if config.get("temperatura") not in TEMPERATURAS:
        problemas.append(f"temperatura debe ser una de {TEMPERATURAS}")
    return problemas


def leer_notebook(nombre):
    return json.loads((RAIZ / "notebooks" / nombre).read_text(encoding="utf-8"))


def salida_de(celda):
    """Texto impreso por una celda de código ya ejecutada."""
    partes = []
    for o in celda.get("outputs", []):
        if "text" in o:
            partes.append("".join(o["text"]))
        elif "text/plain" in o.get("data", {}):
            partes.append("".join(o["data"]["text/plain"]))
    return "\n".join(partes)


class EntregaValida(unittest.TestCase):
    def setUp(self):
        self.entrega = json.loads((RAIZ / "entrega.json").read_text(encoding="utf-8"))

    def test_formato(self):
        self.assertEqual(verificar_entrega(self.entrega), [])

    def test_identificacion(self):
        self.assertEqual(self.entrega["grupo"], "ADAG1")
        self.assertEqual(self.entrega["semana"], 2)

    def test_config_esperada(self):
        self.assertEqual(self.entrega["config"], ESPERADA)


class ValidadorRechazaInvalidas(unittest.TestCase):
    """Casos sencillos con resultado esperado: cada config viola una restricción."""

    def caso(self, **cambios):
        d = {"grupo": "ADAG1", "config": dict(ESPERADA), "semana": 2}
        d["config"].update(cambios)
        return verificar_entrega(d)

    def test_valida(self):
        self.assertEqual(self.caso(), [])

    def test_indice_fuera_de_rango(self):
        self.assertEqual(self.caso(rol=8), ["rol debe ser un entero entre 0 y 7"])
        self.assertEqual(self.caso(cierre=-1), ["cierre debe ser un entero entre 0 y 7"])

    def test_indice_no_entero(self):
        self.assertEqual(self.caso(formato=2.0), ["formato debe ser un entero entre 0 y 7"])
        self.assertEqual(self.caso(estilo=True), ["estilo debe ser un entero entre 0 y 7"])

    def test_temperatura_no_permitida(self):
        self.assertEqual(self.caso(temperatura=0.5), [f"temperatura debe ser una de {TEMPERATURAS}"])

    def test_falta_ranura(self):
        d = {"grupo": "ADAG1", "config": dict(ESPERADA), "semana": 2}
        del d["config"]["estrategia"]
        self.assertEqual(len(verificar_entrega(d)), 2)  # falta el campo y su valor

    def test_semana_como_texto(self):
        d = {"grupo": "ADAG1", "config": dict(ESPERADA), "semana": "2"}
        self.assertEqual(verificar_entrega(d), ["semana debe ser entero"])


class Coherencia(unittest.TestCase):
    """La misma config en entrega.json, en los dos notebooks y en el registro."""

    def setUp(self):
        self.config = json.loads((RAIZ / "entrega.json").read_text(encoding="utf-8"))["config"]

    def test_registro(self):
        with open(RAIZ / "notebooks" / "registro_evaluaciones.csv", encoding="utf-8") as f:
            filas = [x for x in csv.DictReader(f) if float(x["temperatura"]) == 0.0]
        # Misma regla que la sección 8 del notebook: más aciertos; en empate, la primera evaluada.
        mejor = max(filas, key=lambda x: (int(x["aciertos"]), -int(x["n"])))
        elegida = {r: int(mejor[r]) for r in RANURAS}
        self.assertEqual(elegida, {r: self.config[r] for r in RANURAS})
        self.assertEqual(int(mejor["aciertos"]), 17)

    def test_notebook_busqueda(self):
        salidas = "\n".join(salida_de(c) for c in leer_notebook("busqueda.ipynb")["cells"]
                            if c["cell_type"] == "code")
        elegida = re.search(r"CONFIGURACIÓN ELEGIDA: (\{.*?\})", salidas)
        self.assertIsNotNone(elegida, "el notebook de búsqueda no tiene la salida de la sección 8")
        self.assertEqual(ast.literal_eval(elegida.group(1)), self.config)
        self.assertIn('"grupo": "ADAG1"', salidas)

    def test_notebook_evaluacion(self):
        fuente = "\n".join("".join(c["source"]) for c in leer_notebook("evaluacion.ipynb")["cells"])
        self.assertIn('ENTREGA = "entrega.json"', fuente)
        registrada = re.search(r'"config": (\{.*?\})', fuente)
        self.assertEqual(json.loads(registrada.group(1)), self.config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
