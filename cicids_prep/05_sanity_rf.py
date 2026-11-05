"""
Этап 5 (проверка): Random Forest на подготовленных массивах. Нужен, чтобы увидеть, насколько протокол
влияет на оценку (A: случайное, C: блоки по времени) и что массивы собраны без ошибок.
Запуск: python 05_sanity_rf.py data/processed/arrays_time_full.npz [data/processed/arrays_random_full.npz ...]
"""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

res = {}
for path in sys.argv[1:]:
    d = np.load(path, allow_pickle=True)
    names = list(d["class_names"])
    rf = RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=0, class_weight="balanced_subsample")
    rf.fit(d["X_train"], d["y_train"])
    yp = rf.predict(d["X_test"]); yt = d["y_test"]
    labs = [k for k in range(len(names)) if (yt == k).any()]
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labs, zero_division=0)
    tab = pd.DataFrame({"n_test": s, "P": p, "R": r, "F1": f}, index=[names[k] for k in labs])
    tag = path.split("arrays_")[-1].replace(".npz", "")
    res[tag] = tab
    print(f"\n=== {tag}: accuracy {accuracy_score(yt, yp):.4f}, macro-F1 {f.mean():.4f} ===")
    print(tab.round(3).to_string())
if len(res) > 1:
    print("\nF1 по классам:\n", pd.DataFrame({k: v["F1"] for k, v in res.items()}).round(3).to_string())
