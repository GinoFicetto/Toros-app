# Toros 18 · ¿Qué tan bueno va a ser este toro a los 18 meses?

Aplicación web del **Trabajo Práctico 4** de *Aprendizaje Automático y Grandes Datos* (Licenciatura en Agroinformática, UNRaf, 2026).

**Autores:** Gino Ficetto y Bautista Ferrer
**Aplicación publicada:** https://toros-app-unraf.streamlit.app/

## Qué hace

A partir de las DEP que se conocen temprano en un toro (peso al nacer, peso al destete, aptitud materna y circunferencia escrotal) y su año de nacimiento, la aplicación:

1. **Estima la DEP de peso a los 18 meses** (kg), con un rango probable. Modelo: regresión lineal múltiple (TP2).
2. **Da la probabilidad de que el toro esté en el 10 % superior de su raza** a los 18 meses (RANK ≤ 10). Modelo: regresión logística con umbral 0,30 (TP3).
3. **Dice a qué tipo de toro se parece**: crecimiento, materno o generación anterior. Modelo: K-Means con 3 grupos (TP3).

Además:
- Valida las entradas y avisa cuando un valor está fuera de lo observado o la combinación es poco común.
- Muestra dónde cae el toro entre los del sumario y qué aporta cada DEP al resultado.
- Analiza muchos toros juntos desde un archivo CSV y permite descargar los resultados.
- Guarda un registro de los análisis, para ver cómo se usa la aplicación y si los datos que se cargan se alejan de los del entrenamiento.

Cada raza (Braford y Brangus) tiene sus propios modelos, porque las DEP de una raza no se comparan con las de la otra.

## Datos

Sumarios de padres 2026 de la **Asociación Braford Argentina** (1106 toros) y de la **Asociación Argentina de Brangus** (31.º sumario ERBra, 1223 toros), pasados a tabla en el TP1.

## Rendimiento de los modelos

Medido sobre el 20 % de los toros que el modelo no vio al entrenarse (semilla 2026):

| Raza | Kilos: RMSE | Kilos: R² | Top 10 %: AUC | Top 10 % con umbral 0,30: exhaustividad |
|---|---|---|---|---|
| Braford | 10,29 kg | 0,379 | 0,774 | 76 % |
| Brangus | 5,28 kg | 0,660 | 0,886 | 89 % |

Para la aplicación, los modelos se reentrenaron con todos los toros de cada raza, con los mismos hiperparámetros.

## Estructura del repositorio

```
app.py                      la aplicación (Streamlit)
entrenar_modelos.py         entrena y guarda los modelos (reproduce modelos/modelos_toros.joblib)
modelos/modelos_toros.joblib  modelos serializados con joblib (scikit-learn 1.9.1)
datos/toros_braford_brangus.csv  datos limpios (salida del TP1)
ejemplos/toros_ejemplo.csv  archivo de ejemplo para la carga masiva
notebooks/                  cuadernos de los TP1, TP2 y TP3, con sus datos
pruebas/test_app.py         pruebas automáticas de la aplicación
requirements.txt            librerías con versión fija
.streamlit/config.toml      colores y opciones de la aplicación
```

## Cómo correrla en tu computadora

Hace falta Python 3.12 o 3.13.

```bash
git clone https://github.com/GinoFicetto/toros-app.git
cd toros-app
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux o Mac: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Se abre en `http://localhost:8501`.

Para volver a entrenar los modelos (por ejemplo, con un sumario nuevo con las mismas columnas):

```bash
python entrenar_modelos.py
```

Si el archivo de modelos no se puede abrir (por ejemplo, con otra versión de scikit-learn), la aplicación los reentrena sola al iniciar, en unos segundos.

Para correr las pruebas: `python pruebas/test_app.py`.

## Cuadernos

En `notebooks/` están los tres trabajos anteriores, ejecutados y reproducibles (abrirlos desde esa carpeta y usar *Run All*):

- `TP1_exploracion_toros.ipynb`: calidad de datos, limpieza, exploración y elección del toro líder.
- `TP2_regresion_toros.ipynb`: regresión (lineal, k vecinos y Random Forest).
- `TP3_clasificacion_toros.ipynb`: clasificación (logística, k vecinos y Random Forest), clustering (K-Means, Ward y DBSCAN) y ensamble.

## Limitaciones

- Es un primer filtro: no reemplaza la evaluación genética oficial.
- Los modelos se entrenaron con un solo año de sumario (2026); si cambia la evaluación, hay que reentrenarlos.
- En Braford el error típico es de unos 10 kg: el rango probable que muestra la aplicación es ancho, y está bien que lo sea.
- En Streamlit Community Cloud el registro de análisis se borra cuando la aplicación se reinicia.
