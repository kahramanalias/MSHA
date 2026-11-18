# =============================================================================
# What Can Be Known When a Mining Injury Is Reported? Temporal Availability and
# Severity Prediction in 24 Years of MSHA Records (Applied Sciences, 2026)
# Author: Abdulgani Kahraman, Balikesir University
#
# Step 5. 1000-term model with prior-year threshold (Table 6), notification-set (structured fields only)
#   results with prior-year threshold (Table 6), and the scan for field values with
#   P(severe | value) = 1 (Section 3.4). Requires steps 2 and 4.
#
# Run order: 01 -> 02 -> (03) -> 04 -> 05 -> 06 -> 07 -> 08. See README.md.
# Written for Google Colab (each script was run as one cell). Outside Colab the
# scripts use the current folder as BASE; put Accidents.txt there.
# Comments and console messages are partly in Turkish; the code is unchanged
# from the runs reported in the paper except for the data-path lines marked
# "[repo]".
# =============================================================================

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MSHA — EK HAKEM ANALİZLERİ                         [v22-reviewer]       ║
# ║  v21 ve v19 kayıtlarını kullanır (önce onlar çalışmış olmalı).           ║
# ║   C  1000 terim + önceki yıl eşiği (birlikte), S ve B, 2018-2023         ║
# ║   N  yalnız yapısal alanlar (anlatısız) report-time: L3 P/R/F1/AP/NNA    ║
# ║      + aynı modele önceki yıl eşiği                                      ║
# ║   D  denetim 3. adım: report-time alanlarında P(ağır | değer)=1 taraması ║
# ║  Süre: ~30-60 dk. Kopsa aynı hücreyi yeniden çalıştır, devam eder.       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import os, re, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
import lightgbm as lgb

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    BASE = "/content/drive/MyDrive"
except Exception:
    BASE = "."
V7 = os.path.join(BASE, "MSHA_AEJ_v7")
V21 = os.path.join(V7, "pastvocab_v21", "ckpt")
V19 = os.path.join(V7, "sensitivity_v19", "ckpt")
CK = os.path.join(V7, "reviewer_v22", "ckpt"); os.makedirs(CK, exist_ok=True)
SEED, SEVERE, MAX_TRAIN = 20260820, 3, 60_000
RECENT = list(range(2018, 2024))
print("[v22-reviewer] →", CK)

LGB = dict(objective="multiclass", num_class=4, n_estimators=300, learning_rate=0.05,
           num_leaves=63, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
           is_unbalance=True, reg_alpha=0.1, reg_lambda=0.1, random_state=SEED,
           n_jobs=-1, verbose=-1)
# v19 ile aynı sonuç-terimi listesi (1000 terimlik sözlük v19'da bununla seçildi)
OUTCOME_STEMS_V19 = ["fatal", "died", "death", "deceas", "kill", "dead", "doa", "hospital",
    "admitted", "icu", "ambulanc", "airlift", "life flight", "lifeflight", "medevac", "amput",
    "dismember", "paralys", "paraly", "quadripleg", "parapleg", "permanent", "disabilit",
    "disabl", "days away", "days lost", "lost time", "losttime", "restricted duty",
    "restricted work", "light duty", "lt duty", "time off", "off work", "returned to work",
    "return to work", "rtw", "first aid", "firstaid", "no lost", "released", "no treatment",
    "coroner", "autops", "funeral", "morgue"]
RE19 = re.compile("|".join(re.escape(s) for s in OUTCOME_STEMS_V19))

def pload(p):
    with open(p, "rb") as f: return pickle.load(f)
def ck(name, fn):
    p = os.path.join(CK, name + ".pkl")
    if os.path.exists(p):
        try: return pload(p)
        except Exception: os.remove(p)
    t0 = time.perf_counter(); r = fn()
    with open(p + ".tmp", "wb") as f: pickle.dump(r, f)
    os.replace(p + ".tmp", p)
    print(f"  {name:34s} ({time.perf_counter()-t0:.0f}s)")
    return r

dS, featsS, _ = pload(os.path.join(V7, "arm_S", "ckpt", "data.pkl"))
dB, featsB, _ = pload(os.path.join(V7, "arm_B", "ckpt", "data.pkl"))
DATA = {"S": (dS, featsS), "B": (dB, featsB)}

def strat(d, n):
    if len(d) <= n: return d
    return d.groupby("TARGET", group_keys=False).apply(
        lambda g: g.sample(max(1, int(round(n * len(g) / len(d)))), random_state=SEED))

def learn_vocab1000(texts):                       # v19 ile özdeş
    cv = CountVectorizer(ngram_range=(1, 2), min_df=5, max_features=6000, binary=True)
    X = cv.fit_transform(texts.fillna("")); df = np.asarray(X.sum(0)).ravel()
    names = cv.get_feature_names_out()
    return [names[i] for i in np.argsort(-df) if not RE19.search(names[i])][:1000]

def mats(tr, others, feats, vocab):
    num = tr[feats].select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in feats if c not in num]
    maps = {c: {v: i for i, v in enumerate(sorted(tr[c].astype(str).fillna("NA").unique()))} for c in cat}
    tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, vocabulary=vocab).fit(tr["_TXT"]) if vocab else None
    def T(d):
        parts = []
        if num: parts.append(sp.csr_matrix(d[num].apply(pd.to_numeric, errors="coerce").fillna(-1).to_numpy(np.float32)))
        if cat: parts.append(sp.csr_matrix(np.column_stack([d[c].astype(str).fillna("NA").map(maps[c]).fillna(-1).to_numpy(np.float32) for c in cat])))
        if tf is not None: parts.append(tf.transform(d["_TXT"]).astype(np.float32))
        return sp.hstack(parts).tocsr()
    return T(tr), [T(o) for o in others]

def p3_fit(tr, te, feats, vocab):
    Xtr, (Xte,) = mats(tr, [te], feats, vocab)
    return lgb.LGBMClassifier(**LGB).fit(Xtr, tr["TARGET"].values).predict_proba(Xte)[:, 3].astype(np.float32)

def best_thr(yt, p3):
    pr, rc, th = precision_recall_curve(yt == SEVERE, p3)
    f = 2 * pr * rc / np.maximum(pr + rc, 1e-12); i = int(np.nanargmax(f[:-1])); return float(th[i])

def apply_thr(proba, thr):
    yp = np.argmax(proba[:, :3], 1); yp[proba[:, 3] >= thr] = SEVERE; return yp.astype(np.int8)

def mf(r): return f1_score(r["yt"], r["yp"], average="macro", zero_division=0)
def line(label, recs):
    yt = np.concatenate([r["yt"] for r in recs]); yp = np.concatenate([r["yp"] for r in recs])
    tp = int(((yp == SEVERE) & (yt == SEVERE)).sum()); fl = int((yp == SEVERE).sum()); ns = int((yt == SEVERE).sum())
    P = tp / fl if fl else 0.0; R = tp / ns if ns else 0.0; F = 2*P*R/(P+R) if P+R else 0.0
    ap = None
    if all(r.get("p3") is not None for r in recs):
        ap = round(float(average_precision_score(yt == SEVERE, np.concatenate([r["p3"] for r in recs]))), 3)
    return (f"  {label:44s} macroF1={np.mean([mf(r) for r in recs]):.3f}  P={P:.3f} R={R:.3f} F1={F:.3f} "
            f"işaret={fl} kaçan={ns-tp} NNA={(fl/tp if tp else float('nan')):.2f} AP={ap}")
def oracle(label, recs):
    yt = np.concatenate([r["yt"] for r in recs]) == SEVERE; p3 = np.concatenate([r["p3"] for r in recs])
    pr, rc, th = precision_recall_curve(yt, p3); f = 2*pr*rc/np.maximum(pr+rc, 1e-12); i = int(np.nanargmax(f[:-1]))
    tp = int(round(yt.sum()*rc[i])); fl = int(round(tp/pr[i]))
    return f"  {label:44s} ORACLE F1={f[i]:.3f} P={pr[i]:.3f} R={rc[i]:.3f} işaret≈{fl} kaçan={int(yt.sum())-tp} NNA={1/pr[i]:.2f}"

# ── C: 1000 terim + önceki yıl eşiği ───────────────────────────────────────
print("\n=== C: 1000 terim + önceki yıl eşiği ===")
C = {}
for arm in ["S", "B"]:
    d, feats = DATA[arm]
    for y in RECENT:
        base = pload(os.path.join(V19, f"{arm}_C_tfidf1000_{y}.pkl"))      # yıl y, 1000 terim, eğitim < y
        def f(arm=arm, y=y, d=d, feats=feats, base=base):
            tr, va = strat(d[d.YEAR < y - 1], MAX_TRAIN), d[d.YEAR == y - 1]
            thr = best_thr(va["TARGET"].values, p3_fit(tr, va, feats, learn_vocab1000(tr["_TXT"])))
            return {"yt": base["yt"], "yp": apply_thr(base["proba"], thr), "p3": base["p3"], "thr": thr}
        C[(arm, y)] = ck(f"C_{arm}_t1000_prevthr_{y}", f)
        C[(arm, "base", y)] = {"yt": base["yt"], "yp": base["yp"], "p3": base["p3"]}

# ── N: yalnız yapısal alanlar (report-time) ────────────────────────────────
print("\n=== N: yalnız yapısal alanlar ===")
NN, NT = {}, {}
for y in RECENT:
    r = pload(os.path.join(V21, f"N_S_nonarr_{y}.pkl"))                   # v21: eğitim < y, anlatısız
    NN[y] = r
    def f(y=y, r=r):
        tr, va = strat(dS[dS.YEAR < y - 1], MAX_TRAIN), dS[dS.YEAR == y - 1]
        thr = best_thr(va["TARGET"].values, p3_fit(tr, va, featsS, []))
        return {"yt": r["yt"], "yp": apply_thr(r["proba"], thr), "p3": r["p3"], "thr": thr}
    NT[y] = ck(f"N_S_nonarr_prevthr_{y}", f)

# ── D: denetim 3. adım, report-time alanları ───────────────────────────────
print("\n=== D: P(ağır | değer) taraması, report-time alanları ===")
def scan(d, label, min_n=30):
    rows = []
    sev = (d["TARGET"].values == SEVERE)
    for c in featsS:
        v = d[c].astype(str).fillna("NA")
        g = pd.DataFrame({"v": v.values, "s": sev}).groupby("v")["s"].agg(["size", "sum"])
        g = g[g["size"] >= min_n]
        if len(g) == 0: continue
        g["p"] = g["sum"] / g["size"]
        top = g.sort_values(["p", "size"], ascending=False).iloc[0]
        rows.append({"field": c, "n_values_ge_min": len(g), "max_p": round(float(top["p"]), 3),
                     "max_value": str(top.name)[:40], "max_n": int(top["size"]),
                     "n_p_eq_1": int((g["p"] >= 0.9999).sum()), "n_p_ge_0.5": int((g["p"] >= 0.5).sum())})
    out = pd.DataFrame(rows).sort_values("max_p", ascending=False)
    print(f"\n  [{label}] kayıt={len(d):,} ağır oranı={sev.mean():.4f} (değer başına en az {min_n} kayıt)")
    print(out.to_string(index=False))
    return out
D_train = scan(dS[dS.YEAR < 2018], "2000-2017 (eğitim dönemi)")
D_test = scan(dS[dS.YEAR.isin(RECENT)], "2018-2023 (test penceresi)")
D_all = scan(dS, "2000-2023 tümü")

# ══ ÖZET ═══════════════════════════════════════════════════════════════════
print("\n" + "█" * 100 + "\n  ÖZET v22 (2018-2023; macro F1 = yıllık ortalama, L3 havuzlanmış)\n" + "█" * 100)
for arm, n in [("S", "Report-time"), ("B", "Assessment")]:
    print(line(f"{n} 1000-term (v19, referans)", [C[(arm, "base", y)] for y in RECENT]))
    print(line(f"{n} 1000-term + prior-year threshold", [C[(arm, y)] for y in RECENT])
          + f"  eşikler={[round(C[(arm, y)]['thr'], 3) for y in RECENT]}")
    print(oracle(f"{n} 1000-term", [C[(arm, "base", y)] for y in RECENT]))
print(line("Report-time structured only (no narrative)", [NN[y] for y in RECENT]))
print(line("Report-time structured only + prior-year thr", [NT[y] for y in RECENT])
      + f"  eşikler={[round(NT[y]['thr'], 3) for y in RECENT]}")
print(oracle("Report-time structured only", [NN[y] for y in RECENT]))
print("\n[D] P(ağır|değer)=1 olan değer sayısı (n≥30): eğitim", int(D_train.n_p_eq_1.sum()),
      "| test", int(D_test.n_p_eq_1.sum()), "| tümü", int(D_all.n_p_eq_1.sum()))
print("    en yüksek P (tümü):", D_all.iloc[0][["field", "max_value", "max_n", "max_p"]].to_dict())
print(f"\n✅ Bitti. Çıktının TAMAMINI gönder. Kayıtlar: {CK}")
