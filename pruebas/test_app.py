"""Pruebas rápidas de la aplicación. Correr desde la carpeta del repositorio:  python -m pytest pruebas -q
(o sin pytest:  python pruebas/test_app.py)"""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_formulario_braford():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()                         # "Analizar el toro" con los valores típicos
    assert not at.exception
    etiquetas = [m.label for m in at.metric]
    assert "DEP de peso a los 18 meses (estimada)" in etiquetas and "Tipo de toro" in etiquetas


def test_toro_del_sumario_brangus():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Brangus").run()
    at.sidebar.radio[1].set_value("Elegir un toro del sumario 2026").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Valor real" in i.value for i in at.info)


if __name__ == "__main__":
    test_formulario_braford(); test_toro_del_sumario_brangus(); print("pruebas OK")
