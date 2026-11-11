"""
Entrena y guarda los modelos que usa la aplicación (TP4).

Son los modelos elegidos en los TP anteriores, con los mismos datos, variables y semilla:
  - TP2 (regresión):     regresión lineal múltiple -> DEP de peso a los 18 meses (kg)
  - TP3 (clasificación): regresión logística       -> probabilidad de estar en el top 10 % a los 18 meses
  - TP3 (clustering):    K-Means con k = 3         -> perfil del toro (crecimiento, materno o generación anterior)

Para cada raza se hace lo mismo:
  1. Se repite la evaluación del TP (partición 80/20, semilla 2026) para guardar las métricas de prueba.
  2. Se reentrena el modelo final con todos los toros de la raza (los hiperparámetros no cambian).
  3. Se guardan los rangos de cada variable, los residuos fuera de pliegue (para el intervalo de la
     predicción) y un detector de toros raros (bosque de aislamiento, clase 46).

Uso:  python entrenar_modelos.py            (escribe modelos/modelos_toros.joblib)
"""
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
                             precision_score, recall_score, f1_score, accuracy_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEMILLA = 2026
UMBRAL = 0.30                     # umbral de la logística elegido en el TP3 (prioriza no perder toros top)
PREDICTORAS = ["PesoNacer_DEP", "PesoDestete_DEP", "AptitudMaterna_DEP", "CircEscrotal_DEP", "Anio_Nacimiento"]
NOMBRES = {"PesoNacer_DEP": "DEP peso al nacer (kg)", "PesoDestete_DEP": "DEP peso al destete (kg)",
           "AptitudMaterna_DEP": "DEP aptitud materna (kg)", "CircEscrotal_DEP": "DEP circunferencia escrotal (cm)",
           "Anio_Nacimiento": "Año de nacimiento"}
CARPETA = Path(__file__).parent
DATOS = CARPETA / "datos" / "toros_braford_brangus.csv"
SALIDA = CARPETA / "modelos" / "modelos_toros.joblib"


def flujo(modelo):
    """Imputar con la mediana + escalar + modelo, en un solo objeto (sin fuga de información)."""
    return Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler()), ("modelo", modelo)])


def nombrar_grupos(centros):
    """Nombra los 3 grupos de K-Means con el mismo criterio del TP3."""
    viejo = int(centros["Anio_Nacimiento"].idxmin())
    resto = [g for g in range(len(centros)) if g != viejo]
    crec = max(resto, key=lambda g: centros.loc[g, "PesoDestete_DEP"])
    otro = [g for g in resto if g != crec][0]
    return {viejo: "Generación anterior", crec: "Crecimiento", otro: "Materno"}


def entrenar_raza(toros, raza):
    d = toros[toros["Raza"] == raza].reset_index(drop=True)
    X, y_kg, y_top = d[PREDICTORAS], d["Peso18Meses_DEP"], d["top10"]

    # ---- 1. Evaluación, igual que en los TP (para informar en la app y en el artículo)
    Xe, Xp, ye, yp = train_test_split(X, y_kg, test_size=0.2, random_state=SEMILLA)
    lin = flujo(LinearRegression()).fit(Xe, ye); pred = lin.predict(Xp)
    cv_rmse = -cross_val_score(flujo(LinearRegression()), Xe, ye, cv=KFold(5, shuffle=True, random_state=SEMILLA),
                               scoring="neg_root_mean_squared_error")
    met_reg = {"RMSE prueba": float(np.sqrt(mean_squared_error(yp, pred))), "MAE prueba": float(mean_absolute_error(yp, pred)),
               "R² prueba": float(r2_score(yp, pred)), "RMSE CV": float(cv_rmse.mean()), "RMSE CV desvío": float(cv_rmse.std()),
               "n entrenamiento": len(Xe), "n prueba": len(Xp)}

    Xe, Xp, te, tp = train_test_split(X, y_top, test_size=0.2, random_state=SEMILLA, stratify=y_top)
    log = flujo(LogisticRegression(max_iter=1000)).fit(Xe, te); p = log.predict_proba(Xp)[:, 1]
    cv_auc = cross_val_score(flujo(LogisticRegression(max_iter=1000)), Xe, te,
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA), scoring="roc_auc")
    met_cla = {"AUC prueba": float(roc_auc_score(tp, p)), "AUC CV": float(cv_auc.mean()), "AUC CV desvío": float(cv_auc.std()),
               "umbral": UMBRAL, "n prueba": len(Xp), "top en prueba": int(tp.sum())}
    for u in (0.5, UMBRAL):
        s = (p >= u).astype(int)
        met_cla[f"umbral {u}"] = {"exactitud": float(accuracy_score(tp, s)), "precisión": float(precision_score(tp, s)),
                                  "exhaustividad": float(recall_score(tp, s)), "F1": float(f1_score(tp, s))}

    # ---- 2. Modelos finales con todos los toros de la raza
    regresion = flujo(LinearRegression()).fit(X, y_kg)
    clasificador = flujo(LogisticRegression(max_iter=1000)).fit(X, y_top)
    prep = Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]).fit(X)
    Z = prep.transform(X)
    kmeans = KMeans(n_clusters=3, n_init=20, random_state=SEMILLA).fit(Z)
    centros = pd.DataFrame(prep.named_steps["escalar"].inverse_transform(kmeans.cluster_centers_), columns=PREDICTORAS)
    nombres = nombrar_grupos(centros)
    rareza = IsolationForest(n_estimators=200, random_state=SEMILLA).fit(Z)

    # ---- 3. Lo que la app necesita para validar y dar contexto
    # residuos fuera de pliegue: cada toro predicho por un modelo que no lo vio -> intervalo honesto
    residuos = (y_kg - cross_val_predict(flujo(LinearRegression()), X, y_kg, cv=KFold(5, shuffle=True, random_state=SEMILLA))).to_numpy()
    prob_oof = cross_val_predict(flujo(LogisticRegression(max_iter=1000)), X, y_top,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA), method="predict_proba")[:, 1]
    grupos = kmeans.labels_
    rangos = {c: {"min": float(X[c].min()), "max": float(X[c].max()), "p01": float(X[c].quantile(0.01)),
                  "p99": float(X[c].quantile(0.99)), "mediana": float(X[c].median())} for c in PREDICTORAS}
    perfiles = (d.assign(grupo=grupos).groupby("grupo")
                 .agg(toros=("top10", "size"), pct_top=("top10", "mean"), dep18=("Peso18Meses_DEP", "mean"),
                      **{c: (c, "mean") for c in PREDICTORAS}))
    perfiles.index = [nombres[g] for g in perfiles.index]

    return {
        "regresion": regresion, "clasificador": clasificador, "preparar": prep, "kmeans": kmeans,
        "nombres_grupo": nombres, "rareza": rareza, "rareza_ref": rareza.score_samples(Z),
        "metricas_regresion": met_reg, "metricas_clasificacion": met_cla,
        "coef_regresion": dict(zip(PREDICTORAS, regresion.named_steps["modelo"].coef_.tolist())),
        "coef_logistica": dict(zip(PREDICTORAS, clasificador.named_steps["modelo"].coef_[0].tolist())),
        "residuos": residuos, "prob_oof": prob_oof, "rangos": rangos, "perfiles": perfiles,
        "referencia": d[["Nombre"] + PREDICTORAS + ["Peso18Meses_DEP", "Peso18Meses_RANK", "top10"]].assign(grupo=[nombres[g] for g in grupos]),
        "tasa_top": float(y_top.mean()), "n": len(d),
    }


def entrenar_todo(ruta_datos=DATOS):
    toros = pd.read_csv(ruta_datos)
    toros["Nombre"] = toros["Nombre"].astype(str).str.strip()
    toros["top10"] = (toros["Peso18Meses_RANK"] <= 10).astype(int)
    return {"razas": {r: entrenar_raza(toros, r) for r in ["Braford", "Brangus"]},
            "predictoras": PREDICTORAS, "nombres": NOMBRES, "umbral": UMBRAL, "semilla": SEMILLA,
            "version_sklearn": sklearn.__version__, "fecha": date.today().isoformat()}


if __name__ == "__main__":
    paquete = entrenar_todo()
    SALIDA.parent.mkdir(exist_ok=True)
    joblib.dump(paquete, SALIDA, compress=3)
    print(f"Guardado {SALIDA} (scikit-learn {paquete['version_sklearn']})")
    for raza, r in paquete["razas"].items():
        mr, mc = r["metricas_regresion"], r["metricas_clasificacion"]
        print(f"{raza}: {r['n']} toros · lineal RMSE prueba {mr['RMSE prueba']:.2f} kg, R² {mr['R² prueba']:.3f}, "
              f"RMSE CV {mr['RMSE CV']:.2f} ± {mr['RMSE CV desvío']:.2f} · logística AUC prueba {mc['AUC prueba']:.3f}, "
              f"CV {mc['AUC CV']:.3f} ± {mc['AUC CV desvío']:.3f} · exhaustividad con umbral {UMBRAL}: {mc[f'umbral {UMBRAL}']['exhaustividad']:.2f}")
        print("   perfiles:", {k: f"{v['toros']} toros, {v['pct_top']:.0%} top" for k, v in r["perfiles"].iterrows()})
