# =============================================================================
# What Can Be Known When a Mining Injury Is Reported? Temporal Availability and
# Severity Prediction in 24 Years of MSHA Records (Applied Sciences, 2026)
# Author: Abdulgani Kahraman, Balikesir University
#
# Step 2. Sensitivity analyses, 2018-2023: balanced class weights, 1000-term vocabulary learned on each
#   training window, prior-year severe-class threshold, one-hot logistic regression, PR curves.
#   Its 1000-term checkpoints are reused by steps 4-6.
#
# Run order: 01 -> 02 -> (03) -> 04 -> 05 -> 06 -> 07 -> 08. See README.md.
# Written for Google Colab (each script was run as one cell). Outside Colab the
# scripts use the current folder as BASE; put Accidents.txt there.
# Comments and console messages are partly in Turkish; the code is unchanged
# from the runs reported in the paper except for the data-path lines marked
# "[repo]".
# =============================================================================

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MSHA — HAKEM DUYARLILIK ANALİZLERİ (Applied Sciences)  [v19-sensitivity]║
# ║      exec(open('/content/drive/MyDrive/msha_v19_sensitivity_cell.py').read())
# ║                                                                          ║
# ║  Report-time (S) ve Triage (B) setleri, 2018-2023, her yıl <y ile eğitim:║
# ║   A  temel LightGBM (makaledeki ayar; is_unbalance multiclass'ta etkisiz)║
# ║   B  LightGBM class_weight="balanced" (gerçekten etkin ağırlıklandırma)  ║
# ║   C  1000 terimlik TF-IDF (yalnız eğitim penceresinden, sonuç terimleri  ║
# ║      temizlenmiş)                                                        ║
# ║   D  önceki yılda seçilen L3 karar eşiği (eşik y-1'de seçilir, y'de      ║
# ║      uygulanır)                                                          ║
# ║   E  one-hot lojistik regresyon (class_weight="balanced")                ║
# ║  + ağır sınıf PR eğrisi / average precision                              ║
# ║  + triage: amputasyon dışı ağır vakalarda performans                     ║
# ║  + Tablo 5 istatistikleri all_arms_per_year.csv'den                      ║
# ║  ~60-80 dk. Checkpoint'li: kopsa aynı hücreyi tekrar çalıştır.           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os, re, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    ROOT = "/content/drive/MyDrive/MSHA_AEJ_v7"
except Exception:
    ROOT = "./MSHA_AEJ_v7"

CELL_VERSION = "v19-sensitivity"
OUT = os.path.join(ROOT, "sensitivity_v19"); CK = os.path.join(OUT, "ckpt")
os.makedirs(CK, exist_ok=True)
SEED, SEVERE, MAX_TRAIN = 20260820, 3, 60_000
EVAL_YEARS = list(range(2018, 2024))
print(f"[{CELL_VERSION}] → {OUT}")

BASE = dict(objective="multiclass", num_class=4, n_estimators=300, learning_rate=0.05,
            num_leaves=63, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, n_jobs=-1, verbose=-1)
VARIANTS = {
    "A_base":     dict(lgb={**BASE, "is_unbalance": True}, vocab="orig"),
    "B_balanced": dict(lgb={**BASE, "class_weight": "balanced"}, vocab="orig"),
    "C_tfidf1000": dict(lgb={**BASE, "is_unbalance": True}, vocab="train1000"),
}
OUTCOME_STEMS = ["fatal", "died", "death", "deceas", "kill", "dead", "doa", "hospital",
    "admitted", "icu", "ambulanc", "airlift", "life flight", "lifeflight", "medevac", "amput",
    "dismember", "paralys", "paraly", "quadripleg", "parapleg", "permanent", "disabilit",
    "disabl", "days away", "days lost", "lost time", "losttime", "restricted duty",
    "restricted work", "light duty", "lt duty", "time off", "off work", "returned to work",
    "return to work", "rtw", "first aid", "firstaid", "no lost", "released", "no treatment",
    "coroner", "autops", "funeral", "morgue"]
OUTCOME_RE = re.compile("|".join(re.escape(s) for s in OUTCOME_STEMS))

def strat(d, n):
    if len(d) <= n: return d
    return d.groupby("TARGET", group_keys=False).apply(
        lambda g: g.sample(max(1, int(round(n * len(g) / len(d)))), random_state=SEED))

def learn_vocab(texts, k=1000):
    cv = CountVectorizer(ngram_range=(1, 2), min_df=5, max_features=6000, binary=True)
    X = cv.fit_transform(texts.fillna(""))
    df = np.asarray(X.sum(0)).ravel()
    names = cv.get_feature_names_out()
    order = np.argsort(-df)                       # belge frekansına göre (alfabetik DEĞİL)
    keep = [names[i] for i in order if not OUTCOME_RE.search(names[i])]
    return keep[:k]

def build_tree(tr, others, feats, vocab):
    num = tr[feats].select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in feats if c not in num]
    maps = {c: {v: i for i, v in enumerate(sorted(tr[c].astype(str).fillna("NA").unique()))}
            for c in cat}
    tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, vocabulary=vocab).fit(tr["_TXT"])
    def T(d):
        parts = []
        if num:
            parts.append(sp.csr_matrix(d[num].apply(pd.to_numeric, errors="coerce")
                                        .fillna(-1).to_numpy(np.float32)))
        if cat:
            parts.append(sp.csr_matrix(np.column_stack([
                d[c].astype(str).fillna("NA").map(maps[c]).fillna(-1).to_numpy(np.float32)
                for c in cat])))
        parts.append(tf.transform(d["_TXT"]).astype(np.float32))
        return sp.hstack(parts).tocsr()
    return T(tr), [T(o) for o in others]

def build_onehot(tr, te, feats, vocab):
    num = tr[feats].select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in feats if c not in num]
    oh = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
    oh.fit(tr[cat].astype(str).fillna("NA"))
    tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, vocabulary=vocab).fit(tr["_TXT"])
    def T(d):
        parts = [oh.transform(d[cat].astype(str).fillna("NA"))]
        if num:
            parts.append(sp.csr_matrix(d[num].apply(pd.to_numeric, errors="coerce")
                                        .fillna(-1).to_numpy(np.float32)))
        parts.append(tf.transform(d["_TXT"]))
        return sp.hstack(parts).tocsr()
    sc = MaxAbsScaler().fit(T(tr))
    return sc.transform(T(tr)), sc.transform(T(te))

def ckload(name):
    p = os.path.join(CK, name + ".pkl")
    if os.path.exists(p):
        with open(p, "rb") as f: return pickle.load(f)
    return None

def cksave(name, obj):
    p = os.path.join(CK, name + ".pkl")
    with open(p + ".tmp", "wb") as f: pickle.dump(obj, f)
    os.replace(p + ".tmp", p)

def best_thr(yt, p3):
    pr, rc, th = precision_recall_curve(yt == SEVERE, p3)
    f = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    i = int(np.nanargmax(f[:-1])) if len(th) else 0
    return float(th[i]) if len(th) else 0.5, float(f[i])

def apply_thr(proba, thr):
    yp = np.argmax(proba[:, :3], axis=1)
    yp[proba[:, 3] >= thr] = SEVERE
    return yp

def summarise(arm, var, recs):
    yt = np.concatenate([r["yt"] for r in recs]); yp = np.concatenate([r["yp"] for r in recs])
    tp = int(((yp == SEVERE) & (yt == SEVERE)).sum()); fl = int((yp == SEVERE).sum())
    ns = int((yt == SEVERE).sum())
    P = tp / fl if fl else 0.0; R = tp / ns if ns else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    ap = np.nan
    if all("p3" in r for r in recs):
        ap = average_precision_score(yt == SEVERE, np.concatenate([r["p3"] for r in recs]))
    return {"set": {"S": "Report-time", "B": "Triage"}[arm], "variant": var,
            "macro_F1_mean": np.mean([f1_score(r["yt"], r["yp"], average="macro") for r in recs]),
            "L3_prec": P, "L3_rec": R, "L3_F1": F, "flagged": fl, "missed": ns - tp,
            "NNA": (fl / tp) if tp else np.inf, "L3_AP": ap, "n_severe": ns}

summary, pooled = [], {}
for arm in ["S", "B"]:
    with open(os.path.join(ROOT, f"arm_{arm}", "ckpt", "data.pkl"), "rb") as f:
        d, feats, terms = pickle.load(f)
    print(f"\n=== {arm}: {len(feats)} yapısal alan, {len(terms)} narratif terim ===")
    res = {}
    for var, cfg in VARIANTS.items():
        recs = []
        for y in EVAL_YEARS:
            key = f"{arm}_{var}_{y}"
            r = ckload(key)
            if r is None:
                t0 = time.perf_counter()
                tr, te = strat(d[d.YEAR < y], MAX_TRAIN), d[d.YEAR == y]
                vocab = terms if cfg["vocab"] == "orig" else learn_vocab(tr["_TXT"], 1000)
                Xtr, (Xte,) = build_tree(tr, [te], feats, vocab)
                m = lgb.LGBMClassifier(**cfg["lgb"]).fit(Xtr, tr["TARGET"].values)
                pr = m.predict_proba(Xte)
                r = {"yt": te["TARGET"].values.astype(np.int8), "yp": np.argmax(pr, 1).astype(np.int8),
                     "p3": pr[:, 3].astype(np.float32), "proba": pr.astype(np.float32),
                     "nature": te["NATURE_INJURY"].astype(str).values if "NATURE_INJURY" in te else None}
                cksave(key, r)
                print(f"  {var:12s} {y}  macroF1={f1_score(r['yt'], r['yp'], average='macro'):.3f}"
                      f"  ({time.perf_counter()-t0:.0f}s)")
            recs.append(r)
        res[var] = recs
        summary.append(summarise(arm, var, recs))
    pooled[arm] = res["A_base"]

    # D — önceki yılda seçilen eşik
    recs = []
    for y, rA in zip(EVAL_YEARS, res["A_base"]):
        key = f"{arm}_D_thr_{y}"
        r = ckload(key)
        if r is None:
            t0 = time.perf_counter()
            tr, va = strat(d[d.YEAR < y - 1], MAX_TRAIN), d[d.YEAR == y - 1]
            Xtr, (Xva,) = build_tree(tr, [va], feats, terms)
            m = lgb.LGBMClassifier(**VARIANTS["A_base"]["lgb"]).fit(Xtr, tr["TARGET"].values)
            thr, fv = best_thr(va["TARGET"].values, m.predict_proba(Xva)[:, 3])
            r = {"yt": rA["yt"], "yp": apply_thr(rA["proba"], thr).astype(np.int8), "thr": thr}
            cksave(key, r)
            print(f"  D_prev_thr   {y}  eşik={thr:.3f} (y-1 F1={fv:.3f})  ({time.perf_counter()-t0:.0f}s)")
        recs.append(r)
    summary.append(summarise(arm, "D_prev_year_threshold", recs))
    print(f"  D eşikleri: {[round(r['thr'],3) for r in recs]}")

    # E — one-hot lojistik regresyon
    recs = []
    for y in EVAL_YEARS:
        key = f"{arm}_E_logreg_onehot_{y}"
        r = ckload(key)
        if r is None:
            t0 = time.perf_counter()
            tr, te = strat(d[d.YEAR < y], MAX_TRAIN), d[d.YEAR == y]
            Xtr, Xte = build_onehot(tr, te, feats, terms)
            m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, tr["TARGET"].values)
            pr = m.predict_proba(Xte)
            r = {"yt": te["TARGET"].values.astype(np.int8), "yp": np.argmax(pr, 1).astype(np.int8),
                 "p3": pr[:, 3].astype(np.float32)}
            cksave(key, r)
            print(f"  E_logreg_oh  {y}  macroF1={f1_score(r['yt'], r['yp'], average='macro'):.3f}"
                  f"  ({time.perf_counter()-t0:.0f}s)")
        recs.append(r)
    summary.append(summarise(arm, "E_logreg_onehot", recs))

S = pd.DataFrame(summary)
S.to_csv(os.path.join(OUT, "sensitivity_summary.csv"), index=False)
pd.set_option("display.width", 200)
print("\n" + "=" * 110 + "\n  ÖZET (2018-2023; macro F1 = yıllık ortalama, L3 metrikleri havuzlanmış)\n" + "=" * 110)
print(S.round(4).to_string(index=False))

# ── PR eğrisi + kehanet (oracle) eşik üst sınırı ─────────────────────────────
fig, ax = plt.subplots(figsize=(4.8, 3.8))
for arm, col, lab in [("S", "#1f4e79", "Report-time"), ("B", "#c9761b", "Triage")]:
    yt = np.concatenate([r["yt"] for r in pooled[arm]]) == SEVERE
    p3 = np.concatenate([r["p3"] for r in pooled[arm]])
    pr, rc, th = precision_recall_curve(yt, p3)
    ap = average_precision_score(yt, p3)
    f = 2 * pr * rc / np.maximum(pr + rc, 1e-12); i = int(np.nanargmax(f[:-1]))
    print(f"\n  {lab}: AP={ap:.3f} | prevalans={yt.mean():.4f} | ORACLE eşik (havuzlanmış, iyimser): "
          f"F1={f[i]:.3f} P={pr[i]:.3f} R={rc[i]:.3f}")
    for target in (0.25, 0.50, 0.75):
        j = np.where(rc >= target)[0]
        if len(j):
            jj = j[np.argmax(pr[j])]
            print(f"     recall≥{target:.2f} için en yüksek precision = {pr[jj]:.3f}  → NNA ≈ {1/max(pr[jj],1e-9):.1f}")
    ax.plot(rc, pr, color=col, lw=1.5, label=f"{lab} (AP = {ap:.3f})")
ax.axhline(yt.mean(), color="#888", ls=":", lw=1)
ax.set_xlabel("Severe-class recall"); ax.set_ylabel("Severe-class precision")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(frameon=False, loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Figure_PR_severe.{e}"), dpi=600, bbox_inches="tight")
plt.close(fig)

# ── Triage: amputasyon dışı ağır vakalar ─────────────────────────────────────
yt = np.concatenate([r["yt"] for r in pooled["B"]]); yp = np.concatenate([r["yp"] for r in pooled["B"]])
nat = np.concatenate([r["nature"] for r in pooled["B"]])
amp = pd.Series(nat).str.contains("AMPUTATION", case=False).to_numpy()
sev = yt == SEVERE
print("\n" + "=" * 80 + "\n  TRIAGE (A_base) — amputasyon kodu içi/dışı\n" + "=" * 80)
print(f"  ağır vaka: toplam {sev.sum()}, amputasyon {int((sev&amp).sum())}, diğer {int((sev&~amp).sum())}")
print(f"  yakalanan: amputasyon {int((sev&amp&(yp==SEVERE)).sum())}/{int((sev&amp).sum())}, "
      f"diğer {int((sev&~amp&(yp==SEVERE)).sum())}/{int((sev&~amp).sum())}")
m = ~amp; tp = int(((yp == SEVERE) & sev & m).sum()); fl = int(((yp == SEVERE) & m).sum()); ns = int((sev & m).sum())
P = tp / fl if fl else 0; R = tp / ns if ns else 0
print(f"  amputasyon kayıtları çıkarıldığında L3: P={P:.3f} R={R:.3f} "
      f"F1={(2*P*R/(P+R) if P+R else 0):.3f} işaret={fl} doğru={tp} kaçan={ns-tp}")

# ── Tablo 5: all_arms_per_year.csv'den gerçek istatistikler ─────────────────
csv = os.path.join(ROOT, "all_arms_per_year.csv")
if os.path.exists(csv):
    a = pd.read_csv(csv); a = a[a.strategy == "static_expanding"]
    ser = {k: a[a.arm == k].sort_values("year").set_index("year").macro_f1 for k in ["S", "B", "L"]}
    rng = np.random.default_rng(SEED)
    print("\n" + "=" * 80 + "\n  TABLO 5 (gerçek yıllık seriler)\n" + "=" * 80)
    for k, n in [("S", "Report-time"), ("B", "Triage"), ("L", "Post-closure")]:
        print(f"  {n:13s} mean={ser[k].mean():.4f} SD={ser[k].std(ddof=1):.4f} n={len(ser[k])}")
    for k, n in [("B", "Triage"), ("L", "Post-closure")]:
        yrs = ser["S"].index.intersection(ser[k].index)
        dd = (ser[k].loc[yrs] - ser["S"].loc[yrs]).to_numpy()
        bs = np.array([rng.choice(dd, len(dd), replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5]); _, p = stats.wilcoxon(dd)
        print(f"  {n} − Report-time: Δ={dd.mean():.4f} 95% CI [{lo:.4f}, {hi:.4f}] Wilcoxon p={p:.1e}")
    print("  yıllık S:", ", ".join(f"{v:.4f}" for v in ser["S"]))
    print("  yıllık B:", ", ".join(f"{v:.4f}" for v in ser["B"]))
    print("  yıllık L:", ", ".join(f"{v:.4f}" for v in ser["L"]))
else:
    print(f"\n  ⚠ {csv} yok — Tablo 5 kontrolü atlandı")

print(f"\n✅ Bitti. Çıktının TAMAMINI kopyalayıp gönder. Şekil: {OUT}/Figure_PR_severe.png")
