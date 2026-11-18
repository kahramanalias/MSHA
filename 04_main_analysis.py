# =============================================================================
# What Can Be Known When a Mining Injury Is Reported? Temporal Availability and
# Severity Prediction in 24 Years of MSHA Records (Applied Sciences, 2026)
# Author: Abdulgani Kahraman, Balikesir University
#
# Step 4. Main analysis with the vocabulary selected only from past records: expanding-window runs
#   for all feature sets 2006-2023, four learners (Table 5), sensitivity analyses (Table 6),
#   field-by-field attribution (Table 9, Figure 5), categorical encoding (Table S4), narrative
#   contribution (Section 2.2), rolling windows and drift detectors (Section 3.6), amputation
#   analysis (Section 3.4). Long run (4-6 h); checkpointed, rerun to resume.
#
# Run order: 01 -> 02 -> (03) -> 04 -> 05 -> 06 -> 07 -> 08. See README.md.
# Written for Google Colab (each script was run as one cell). Outside Colab the
# scripts use the current folder as BASE; put Accidents.txt there.
# Comments and console messages are partly in Turkish; the code is unchanged
# from the runs reported in the paper except for the data-path lines marked
# "[repo]".
# =============================================================================

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MSHA — TÜM ANALİZ, SÖZLÜK YALNIZCA GEÇMİŞTEN       [v21-pastvocab-full] ║
# ║      exec(open('/content/drive/MyDrive/msha_v21_pastvocab_full_cell.py').read())
# ║                                                                          ║
# ║  Her değerlendirme yılı y için 100 terimlik anlatı sözlüğü YALNIZCA      ║
# ║  y'den önceki kayıtlardan seçilir. Makaledeki bütün tablo ve şekiller    ║
# ║  bu koşulardan yeniden üretilir:                                         ║
# ║   A  ana koşu S/B/L 2006-2023 (Tablo 3, 6, Şekil 1-2, istatistikler)     ║
# ║   K  kayma dedektörleri + yıllık hata oranı (Bölüm 3.4)                  ║
# ║   T4 dört öğrenici S/B 2018-2023 (Tablo 4)                               ║
# ║   T5 duyarlılık analizleri S/B 2018-2023 (Tablo 5)                       ║
# ║   T7 alan alan ekleme 2018-2023 (Tablo 7, Şekil 3)                       ║
# ║   S4 kategorik kodlama (Tablo S4) · N anlatı katkısı · W kayan pencere   ║
# ║  Uzun: ~4-6 saat. Her adım Drive'a ayrı kaydedilir; kopsa aynı satırı    ║
# ║  yeniden çalıştır, kaldığı yerden devam eder. En sonda ÖZET basılır.     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import sys, subprocess
try:
    import river  # noqa
except ImportError:
    try: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "river"])
    except Exception: pass

import os, re, json, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (f1_score, balanced_accuracy_score, average_precision_score,
                             precision_recall_curve, precision_recall_fscore_support, confusion_matrix)
import lightgbm as lgb

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    BASE = "/content/drive/MyDrive"
except Exception:
    BASE = "."
V7 = os.path.join(BASE, "MSHA_AEJ_v7")
OUT = os.path.join(V7, "pastvocab_v21"); CK = os.path.join(OUT, "ckpt"); os.makedirs(CK, exist_ok=True)
V19 = os.path.join(V7, "sensitivity_v19", "ckpt"); V20 = os.path.join(V7, "vocab_v20", "ckpt")
SEED, SEVERE, MAX_TRAIN, N_TFIDF = 20260820, 3, 60_000, 100
ALL_YEARS = list(range(2006, 2024)); RECENT = list(range(2018, 2024))
print("[v21-pastvocab-full] →", OUT)

OUTCOME_STEMS = ["fatal", "died", "death", "deceas", "kill", "dead", "doa", "coroner",
    "autops", "expired", "pronounced", "hospital", "hosp", "admitted", "icu", "ambulanc",
    "airlift", "air lift", "life flight", "lifeflight", "medevac", "med evac", "transported",
    "transport to", "emergency room", "er for", "trauma center", "amput", "amp of",
    "dismember", "paraly", "parapleg", "quadripleg", "permanent", "perm partial",
    "perm total", "disabilit", "disabl", "loss of use", "prosthe", "days away", "days lost",
    "day lost", "lost time", "losttime", "lost day", "lost work", "time off", "off work",
    "off duty", "missed work", "restricted duty", "restricted work", "restricted activity",
    "light duty", "lt duty", "lite duty", "modified duty", "restrict", "rtw",
    "return to work", "returned to work", "released to", "first aid", "firstaid",
    "first-aid", "no lost", "no treatment", "released", "no further", "refused treatment",
    "minor injury"]
OUT_RE = re.compile("|".join(map(re.escape, OUTCOME_STEMS)))
LGB = dict(objective="multiclass", num_class=4, n_estimators=300, learning_rate=0.05,
           num_leaves=63, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
           is_unbalance=True, reg_alpha=0.1, reg_lambda=0.1, random_state=SEED,
           n_jobs=-1, verbose=-1)
POST = ["SCHEDULE_CHARGE", "DAYS_LOST", "DAYS_RESTRICT", "TRANS_TERM", "RETURN_TO_WORK_DT",
        "IMMED_NOTIFY", "IMMED_NOTIFY_CD", "INVEST_BEGIN_DT"]

def pload(p):
    with open(p, "rb") as f: return pickle.load(f)
def ck(name, fn):
    p = os.path.join(CK, name + ".pkl")
    if os.path.exists(p):
        try: return pload(p)
        except Exception: os.remove(p)
    t0 = time.perf_counter(); r = fn()
    if r is None: return None
    with open(p + ".tmp", "wb") as f: pickle.dump(r, f)
    os.replace(p + ".tmp", p)
    print(f"  {name:38s} ({time.perf_counter()-t0:.0f}s)")
    return r

dS, featsS, _ = pload(os.path.join(V7, "arm_S", "ckpt", "data.pkl"))
dB, featsB, _ = pload(os.path.join(V7, "arm_B", "ckpt", "data.pkl"))
dL, featsL, _ = pload(os.path.join(V7, "arm_L", "ckpt", "data.pkl"))
assert len(dS) == len(dB) == len(dL)
DATA = {"S": (dS, featsS), "B": (dB, featsB), "L": (dL, featsL)}
print(f"  kayıt {len(dS):,} | alan S {len(featsS)} B {len(featsB)} L {len(featsL)}")

def strat(d, n):
    if len(d) <= n: return d
    return d.groupby("TARGET", group_keys=False).apply(
        lambda g: g.sample(max(1, int(round(n * len(g) / len(d)))), random_state=SEED))

# ── yıllık sözlük: YALNIZCA y'den önceki kayıtlar ───────────────────────────
def vocab_year(y):
    def make():
        cv = CountVectorizer(ngram_range=(1, 2), min_df=20, max_features=2000,
                             stop_words="english", binary=True)
        M = cv.fit_transform(dS.loc[dS.YEAR < y, "_TXT"])
        nm = np.array(cv.get_feature_names_out()); dfq = np.asarray(M.sum(0)).ravel()
        ranked = list(nm[np.argsort(-dfq)])
        raw = ranked[:N_TFIDF]
        return {"clean": [t for t in ranked if not OUT_RE.search(t)][:N_TFIDF], "raw": raw,
                "n_flag_vocab": int(sum(bool(OUT_RE.search(t)) for t in ranked)),
                "n_flag_top100": int(sum(bool(OUT_RE.search(t)) for t in raw))}
    return ck(f"vocab_{y}", make)
def V(arm, y, raw=False):
    v = vocab_year(y); return v["raw"] if (raw or arm == "L") else v["clean"]

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

def lgbm(tr, te, feats, vocab, **kw):
    Xtr, (Xte,) = mats(tr, [te], feats, vocab)
    return lgb.LGBMClassifier(**{**LGB, **kw}).fit(Xtr, tr["TARGET"].values).predict_proba(Xte).astype(np.float32)

def pack(te, proba):
    yt = te["TARGET"].values.astype(np.int8); yp = np.argmax(proba, 1).astype(np.int8)
    return {"yt": yt, "yp": yp, "proba": proba, "p3": proba[:, 3]}

# ── A: ana koşu ────────────────────────────────────────────────────────────
print("\n=== A: ana koşu (genişleyen pencere, sözlük yalnızca geçmişten) ===")
A = {}
for arm in ["S", "B", "L"]:
    d, feats = DATA[arm]
    for y in ALL_YEARS:
        def f(arm=arm, y=y, d=d, feats=feats):
            te = d[d.YEAR == y]
            v20 = os.path.join(V20, f"{arm}_{y}.pkl")
            if arm == "S" and y < 2018 and os.path.exists(v20):   # v20 ile özdeş koşu
                r = pload(v20); return {"yt": r["yt"], "yp": r["yp"], "p3": r["p3"], "proba": None}
            return pack(te, lgbm(strat(d[d.YEAR < y], MAX_TRAIN), te, feats, V(arm, y)))
        A[(arm, y)] = ck(f"A_{arm}_{y}", f)

# ── K: hata akışı ve kayma dedektörleri (report-time) ──────────────────────
print("\n=== K: kayma dedektörleri ===")
def detectors_run():
    out = {}
    try:
        import river.drift as rd
        try: import river.drift.binary as rb
        except Exception: rb = None
    except Exception:
        print("  ⚠ river kurulamadı; dedektör adımı atlandı (yeniden çalıştırınca tekrar denenir)")
        return None
    def find(*c):
        for mod in (rd, rb):
            if mod is None: continue
            for w in c:
                for n in dir(mod):
                    if n.lower().replace("_", "") == w.lower().replace("_", ""): return getattr(mod, n)
    specs = [("ADWIN", ["ADWIN"], "float", dict(delta=0.002)), ("DDM", ["DDM"], "int", {}),
             ("EDDM", ["EDDM"], "int", {}), ("HDDM-A", ["HDDM_A", "HDDMA"], "int", {}),
             ("HDDM-W", ["HDDM_W", "HDDMW"], "int", {}), ("Page-Hinkley", ["PageHinkley"], "float", {}),
             ("KSWIN", ["KSWIN"], "float", dict(seed=SEED))]
    avail = []
    for lab, c, kind, kw in specs:
        cls = find(*c)
        if cls is None: continue
        try: cls(**kw)
        except TypeError: kw = {}
        avail.append((lab, cls, kind, kw))
    for arm in ["S", "B", "L"]:
        st = {y: (A[(arm, y)]["yp"] != A[(arm, y)]["yt"]).astype(np.int8) for y in ALL_YEARS}
        def run(perm=None, adwin_delta=None):
            rng = np.random.default_rng(perm) if perm is not None else None
            dets = {l: (cls(**({**kw, "delta": adwin_delta} if (adwin_delta and l == "ADWIN") else kw)), k)
                    for l, cls, k, kw in avail}
            hits = {l: [] for l in dets}
            for y in ALL_YEARS:
                e = st[y] if rng is None else st[y][rng.permutation(len(st[y]))]
                for l, (dd, k) in dets.items():
                    fired = False
                    for v in e:
                        dd.update(float(v) if k == "float" else int(v))
                        if getattr(dd, "drift_detected", False): fired = True
                    if fired: hits[l].append(y)
            return hits
        base = run(); perms = [run(perm=SEED + k) for k in range(3)]
        agree = sorted({y for a in base for b in base if a < b for y in set(base[a]) & set(base[b])})
        stable = {l: sum(p[l] == base[l] for p in perms) for l in base}
        sweep = {dl: run(adwin_delta=dl)["ADWIN"] for dl in [0.0005, 0.001, 0.002, 0.005, 0.01]} if "ADWIN" in base else {}
        er = np.array([st[y].mean() for y in ALL_YEARS])
        slope, icpt = np.polyfit(ALL_YEARS, er, 1)
        r2 = 1 - ((er - (slope * np.array(ALL_YEARS) + icpt)) ** 2).sum() / ((er - er.mean()) ** 2).sum()
        out[arm] = {"alarms": base, "years_two_or_more_agree": agree, "perm_stable_of_3": stable,
                    "adwin_delta_sweep": sweep, "err_first": float(er[0]), "err_last": float(er[-1]),
                    "slope_per_decade": float(slope * 10), "linear_r2": float(r2)}
    return out
K = ck("K_detectors", detectors_run)

# ── T4: dört öğrenici ──────────────────────────────────────────────────────
print("\n=== T4: öğreniciler (2018-2023) ===")
def make(name):
    if name == "RF": return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                                   class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    if name == "HistGB": return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
                                                               min_samples_leaf=30, l2_regularization=0.1,
                                                               class_weight="balanced", random_state=SEED)
    return LogisticRegression(max_iter=2000, class_weight="balanced")
T4 = {}
for arm in ["S", "B"]:
    d, feats = DATA[arm]
    for name in ["RF", "HistGB", "LogReg"]:
        for y in RECENT:
            def f(arm=arm, name=name, y=y, d=d, feats=feats):
                tr, te = strat(d[d.YEAR < y], MAX_TRAIN), d[d.YEAR == y]
                Xtr, (Xte,) = mats(tr, [te], feats, V(arm, y))
                if name == "HistGB": Xtr, Xte = Xtr.toarray(), Xte.toarray()
                m = make(name).fit(Xtr, tr["TARGET"].values)
                return {"yt": te["TARGET"].values.astype(np.int8), "yp": m.predict(Xte).astype(np.int8)}
            T4[(arm, name, y)] = ck(f"T4_{arm}_{name}_{y}", f)

# ── T5: duyarlılık ─────────────────────────────────────────────────────────
print("\n=== T5: duyarlılık analizleri (2018-2023) ===")
def best_thr(yt, p3):
    pr, rc, th = precision_recall_curve(yt == SEVERE, p3)
    f = 2 * pr * rc / np.maximum(pr + rc, 1e-12); i = int(np.nanargmax(f[:-1])); return float(th[i])
T5 = {}
for arm in ["S", "B"]:
    d, feats = DATA[arm]
    for y in RECENT:
        te = d[d.YEAR == y]
        T5[(arm, "balanced", y)] = ck(f"T5_{arm}_balanced_{y}", lambda arm=arm, y=y, d=d, feats=feats, te=te:
            pack(te, lgbm(strat(d[d.YEAR < y], MAX_TRAIN), te, feats, V(arm, y), class_weight="balanced", is_unbalance=False)))
        def fD(arm=arm, y=y, d=d, feats=feats):
            va = d[d.YEAR == y - 1]
            p3 = lgbm(strat(d[d.YEAR < y - 1], MAX_TRAIN), va, feats, V(arm, y - 1))[:, 3]
            thr = best_thr(va["TARGET"].values, p3)
            pr = A[(arm, y)]["proba"]; yp = np.argmax(pr[:, :3], 1); yp[pr[:, 3] >= thr] = SEVERE
            return {"yt": A[(arm, y)]["yt"], "yp": yp.astype(np.int8), "thr": thr}
        T5[(arm, "prev_thr", y)] = ck(f"T5_{arm}_prevthr_{y}", fD)
        def fE(arm=arm, y=y, d=d, feats=feats, te=te):
            tr = strat(d[d.YEAR < y], MAX_TRAIN)
            num = tr[feats].select_dtypes(include=[np.number]).columns.tolist(); cat = [c for c in feats if c not in num]
            oh = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20).fit(tr[cat].astype(str).fillna("NA"))
            tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, vocabulary=V(arm, y)).fit(tr["_TXT"])
            def T(dd):
                parts = [oh.transform(dd[cat].astype(str).fillna("NA"))]
                if num: parts.append(sp.csr_matrix(dd[num].apply(pd.to_numeric, errors="coerce").fillna(-1).to_numpy(np.float32)))
                parts.append(tf.transform(dd["_TXT"])); return sp.hstack(parts).tocsr()
            sc = MaxAbsScaler().fit(T(tr))
            m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(T(tr)), tr["TARGET"].values)
            return pack(te, m.predict_proba(sc.transform(T(te))).astype(np.float32))
        T5[(arm, "logreg_onehot", y)] = ck(f"T5_{arm}_logregoh_{y}", fE)
        p = os.path.join(V19, f"{arm}_C_tfidf1000_{y}.pkl")      # v19: sözlük zaten yalnız eğitim penceresinden
        if os.path.exists(p): T5[(arm, "tfidf1000", y)] = pload(p)

# ── T7: alan alan ekleme (alanlar doğal sütun sırasında) ───────────────────
print("\n=== T7: alan alan ekleme (2018-2023) ===")
CLIN = [c for c in featsB if c not in featsS]
CONDS = {"+ DAYS_LOST": ["DAYS_LOST"], "+ SCHEDULE_CHARGE": ["SCHEDULE_CHARGE"],
         "+ DAYS_RESTRICT": ["DAYS_RESTRICT"], "+ TRANS_TERM": ["TRANS_TERM"],
         "+ IMMED_NOTIFY (both forms)": ["IMMED_NOTIFY", "IMMED_NOTIFY_CD"],
         "+ RETURN_TO_WORK_DT": ["RETURN_TO_WORK_DT"], "+ INVEST_BEGIN_DT": ["INVEST_BEGIN_DT"],
         "+ clinical (triage) fields": CLIN,
         "+ all post-closure and severity-triggered fields": [c for c in POST if c in featsL],
         "+ unfiltered narrative": []}
T7 = {}
for cname, extra in CONDS.items():
    want = set(featsS) | set(extra)
    feats = [c for c in featsL if c in want]
    for y in RECENT:
        if cname == "+ clinical (triage) fields" and feats == list(featsB):
            T7[(cname, y)] = A[("B", y)]; continue
        te = dL[dL.YEAR == y]
        T7[(cname, y)] = ck(f"T7_{re.sub('[^A-Za-z]', '', cname)}_{y}", lambda feats=feats, y=y, te=te, cname=cname:
            pack(te, lgbm(strat(dL[dL.YEAR < y], MAX_TRAIN), te, feats, V("S", y, raw=(cname == "+ unfiltered narrative")))))

# ── S4: kategorik kodlama (native) ─────────────────────────────────────────
print("\n=== S4: native kategorik (2018-2023) ===")
S4 = {}
for arm in ["S", "B", "L"]:
    d, feats = DATA[arm]
    for y in RECENT:
        def f(arm=arm, y=y, d=d, feats=feats):
            tr, te = strat(d[d.YEAR < y], MAX_TRAIN), d[d.YEAR == y]
            num = tr[feats].select_dtypes(include=[np.number]).columns.tolist(); cat = [c for c in feats if c not in num]
            maps = {c: {v: i for i, v in enumerate(sorted(tr[c].astype(str).fillna("NA").unique()))} for c in cat}
            voc = V(arm, y); tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, vocabulary=voc).fit(tr["_TXT"])
            def T(dd):
                X = dd[num].apply(pd.to_numeric, errors="coerce").fillna(-1).reset_index(drop=True)
                C = pd.DataFrame({c: dd[c].astype(str).fillna("NA").map(maps[c]).fillna(-1).astype(int).values for c in cat})
                TT = pd.DataFrame(tf.transform(dd["_TXT"]).toarray().astype(np.float32), columns=[f"t{i}" for i in range(len(voc))])
                X = pd.concat([X, C, TT], axis=1)
                for c in cat: X[c] = X[c].astype("category")
                return X
            m = lgb.LGBMClassifier(**LGB).fit(T(tr), tr["TARGET"].values, categorical_feature=cat)
            return {"yt": te["TARGET"].values.astype(np.int8), "yp": m.predict(T(te)).astype(np.int8)}
        S4[(arm, y)] = ck(f"S4_{arm}_{y}", f)

# ── N: anlatı katkısı (report-time, 2018-2023) ─────────────────────────────
print("\n=== N: anlatısız model (2018-2023) ===")
NN = {}
for y in RECENT:
    te = dS[dS.YEAR == y]
    NN[y] = ck(f"N_S_nonarr_{y}", lambda y=y, te=te: pack(te, lgbm(strat(dS[dS.YEAR < y], MAX_TRAIN), te, featsS, [])))

# ── W: kayan pencereler (report-time, 2006-2023) ───────────────────────────
print("\n=== W: kayan pencereler (report-time) ===")
W = {}
for w in [3, 5, 10]:
    for y in ALL_YEARS:
        te = dS[dS.YEAR == y]
        W[(w, y)] = ck(f"W_{w}_{y}", lambda w=w, y=y, te=te:
            pack(te, lgbm(strat(dS[(dS.YEAR >= y - w) & (dS.YEAR < y)], MAX_TRAIN), te, featsS, V("S", y))))

# ══ ÖZET ═══════════════════════════════════════════════════════════════════
def mf(r): return f1_score(r["yt"], r["yp"], average="macro", zero_division=0)
def sev(recs):
    yt = np.concatenate([r["yt"] for r in recs]); yp = np.concatenate([r["yp"] for r in recs])
    tp = int(((yp == SEVERE) & (yt == SEVERE)).sum()); fl = int((yp == SEVERE).sum()); ns = int((yt == SEVERE).sum())
    P = tp / fl if fl else 0.0; R = tp / ns if ns else 0.0
    return dict(P=round(P, 3), R=round(R, 3), F1=round(2*P*R/(P+R), 3) if P+R else 0.0, flagged=fl, missed=ns-tp,
                NNA=round(fl/tp, 2) if tp else None, n_severe=ns)
def ap(recs):
    if any(r.get("p3") is None for r in recs): return None
    return round(float(average_precision_score(np.concatenate([r["yt"] for r in recs]) == SEVERE,
                                               np.concatenate([r["p3"] for r in recs]))), 3)
def line(label, recs):
    s = sev(recs); a = ap(recs)
    return f"  {label:46s} macroF1={np.mean([mf(r) for r in recs]):.3f}  P={s['P']:.3f} R={s['R']:.3f} F1={s['F1']:.3f} işaret={s['flagged']} kaçan={s['missed']} NNA={s['NNA']} AP={a}"
rng = np.random.default_rng(SEED)
def cmp(a, b):
    dd = np.asarray(b) - np.asarray(a)
    bs = np.array([rng.choice(dd, len(dd), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5]); _, p = stats.wilcoxon(dd)
    return f"Δ={dd.mean():+.4f} 95%CI [{lo:+.4f}, {hi:+.4f}] Wilcoxon p={p:.1e} d_z={dd.mean()/dd.std(ddof=1):.1f}"

print("\n" + "█" * 100 + "\n  ÖZET — sözlük yalnızca geçmişten\n" + "█" * 100)
vv = [vocab_year(y) for y in ALL_YEARS]
print(f"\n[Sözlük] ilk 100'deki işaretli terim sayısı (yıllar): {[v['n_flag_top100'] for v in vv]}")
print(f"         tüm 2000 aday içinde işaretli terim (min-max): {min(v['n_flag_vocab'] for v in vv)}-{max(v['n_flag_vocab'] for v in vv)}")

print("\n[Tablo 6] 2006-2023")
ser = {}
for arm, n in [("S", "Report-time"), ("B", "Triage"), ("L", "Outcome-informed")]:
    recs = [A[(arm, y)] for y in ALL_YEARS]; ser[arm] = [mf(r) for r in recs]
    ba = np.mean([balanced_accuracy_score(r["yt"], r["yp"]) for r in recs])
    sr = np.mean([((r["yp"] == SEVERE) & (r["yt"] == SEVERE)).sum() / max((r["yt"] == SEVERE).sum(), 1) for r in recs])
    print(f"  {n:17s} macroF1={np.mean(ser[arm]):.4f} SD={np.std(ser[arm], ddof=1):.4f} balAcc={ba:.4f} severeRecall(ort)={sr:.4f}")
print("  Triage − Report-time:", cmp(ser["S"], ser["B"]))
print("  Outcome-informed − Report-time:", cmp(ser["S"], ser["L"]))
for arm in "SBL": print(f"  yıllık {arm}: " + ", ".join(f"{v:.4f}" for v in ser[arm]))

print("\n[Tablo 3 / Şekil 1] report-time, 2006-2023 havuzlanmış")
yt = np.concatenate([A[("S", y)]["yt"] for y in ALL_YEARS]); yp = np.concatenate([A[("S", y)]["yp"] for y in ALL_YEARS])
P, R, F, Sp = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2, 3], zero_division=0)
for k in range(4): print(f"  L{k}: P={P[k]:.3f} R={R[k]:.3f} F1={F[k]:.3f} n={Sp[k]}")
print(f"  macro: P={P.mean():.3f} R={R.mean():.3f} F1={F.mean():.3f} n={Sp.sum()}")
cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3]); print("  karışıklık (satır=gerçek):\n", cm)
print("  satır-normalize:\n", np.round(cm / cm.sum(1, keepdims=True), 2))

print("\n[Tablo 4] 2018-2023")
for arm, n in [("S", "Report-time"), ("B", "Triage")]:
    print(line(f"{n} LightGBM", [A[(arm, y)] for y in RECENT]))
    for nm in ["RF", "HistGB", "LogReg"]: print(line(f"{n} {nm}", [T4[(arm, nm, y)] for y in RECENT]))

print("\n[Tablo 5] 2018-2023")
for arm, n in [("S", "Report-time"), ("B", "Triage")]:
    print(line(f"{n} LightGBM no reweighting", [A[(arm, y)] for y in RECENT]))
    print(line(f"{n} LightGBM balanced", [T5[(arm, 'balanced', y)] for y in RECENT]))
    if all((arm, "tfidf1000", y) in T5 for y in RECENT): print(line(f"{n} LightGBM 1000-term (v19)", [T5[(arm, 'tfidf1000', y)] for y in RECENT]))
    print(line(f"{n} LightGBM prior-year threshold", [T5[(arm, 'prev_thr', y)] for y in RECENT]) + f"  eşikler={[round(T5[(arm,'prev_thr',y)]['thr'],3) for y in RECENT]}")
    print(line(f"{n} LogReg one-hot", [T5[(arm, 'logreg_onehot', y)] for y in RECENT]))
    ytp = np.concatenate([A[(arm, y)]["yt"] for y in RECENT]) == SEVERE; p3 = np.concatenate([A[(arm, y)]["p3"] for y in RECENT])
    pr, rc, th = precision_recall_curve(ytp, p3); f = 2*pr*rc/np.maximum(pr+rc, 1e-12); i = int(np.nanargmax(f[:-1]))
    print(f"  {n} ORACLE eşik: F1={f[i]:.3f} P={pr[i]:.3f} R={rc[i]:.3f} kaçan={int(round(ytp.sum()*(1-rc[i])))} NNA={1/pr[i]:.2f} | prevalans={ytp.mean():.4f}")
    for t in (0.25, 0.5, 0.75):
        j = np.where(rc >= t)[0]; jj = j[np.argmax(pr[j])]
        print(f"     recall≥{t}: en yüksek precision={pr[jj]:.3f} → NNA≈{1/max(pr[jj],1e-9):.1f}")
nat = np.concatenate([dB[dB.YEAR == y]["NATURE_INJURY"].astype(str).values for y in RECENT])
ytB = np.concatenate([A[("B", y)]["yt"] for y in RECENT]); ypB = np.concatenate([A[("B", y)]["yp"] for y in RECENT])
amp = pd.Series(nat).str.contains("AMPUTATION", case=False).to_numpy(); sv = ytB == SEVERE
m_ = ~amp; tp = int(((ypB == SEVERE) & sv & m_).sum()); fl = int(((ypB == SEVERE) & m_).sum()); ns = int((sv & m_).sum())
Pn, Rn = (tp/fl if fl else 0), (tp/ns if ns else 0)
print(f"  Triage amputasyon: ağır {sv.sum()} (amp {int((sv&amp).sum())}); yakalanan amp {int((sv&amp&(ypB==SEVERE)).sum())}, diğer {int((sv&~amp&(ypB==SEVERE)).sum())}/{int((sv&~amp).sum())}")
print(f"  Triage amputasyon hariç: P={Pn:.3f} R={Rn:.3f} F1={(2*Pn*Rn/(Pn+Rn) if Pn+Rn else 0):.3f} işaret={fl} kaçan={ns-tp} NNA={(fl/tp if tp else float('nan')):.2f}")

print("\n[Tablo 7] 2018-2023 (macroF1 ve ağır duyarlılık = yıllık ortalama)")
base = np.mean([mf(A[("S", y)]) for y in RECENT])
bsr = np.mean([((A[("S",y)]["yp"]==SEVERE)&(A[("S",y)]["yt"]==SEVERE)).sum()/max((A[("S",y)]["yt"]==SEVERE).sum(),1) for y in RECENT])
print(f"  {'Report-time baseline':50s} macroF1={base:.3f}  Δ=  –    severeRecall={bsr:.3f}")
for cname in CONDS:
    recs = [T7[(cname, y)] for y in RECENT]; m = np.mean([mf(r) for r in recs])
    sr = np.mean([((r["yp"]==SEVERE)&(r["yt"]==SEVERE)).sum()/max((r["yt"]==SEVERE).sum(),1) for r in recs])
    print(f"  {cname:50s} macroF1={m:.3f}  Δ={m-base:+.3f}  severeRecall={sr:.3f}")

print("\n[Tablo S4] 2018-2023 havuzlanmış")
for arm in "SBL":
    for lab, recs in [("integer", [A[(arm, y)] for y in RECENT]), ("native", [S4[(arm, y)] for y in RECENT])]:
        yt = np.concatenate([r["yt"] for r in recs]); yp = np.concatenate([r["yp"] for r in recs])
        P, R, F, _ = precision_recall_fscore_support(yt, yp, labels=[0,1,2,3], zero_division=0)
        print(f"  {arm} {lab:8s} macroF1(havuz)={f1_score(yt,yp,average='macro'):.4f} balAcc={balanced_accuracy_score(yt,yp):.4f} L3 P={P[3]:.3f} R={R[3]:.3f} F1={F[3]:.3f}")

print("\n[Anlatı katkısı] report-time 2018-2023: anlatılı "
      f"{np.mean([mf(A[('S',y)]) for y in RECENT]):.4f} | anlatısız {np.mean([mf(NN[y]) for y in RECENT]):.4f} | fark "
      f"{np.mean([mf(A[('S',y)]) for y in RECENT]) - np.mean([mf(NN[y]) for y in RECENT]):+.4f}")

print("\n[Kayan pencereler] report-time 2006-2023, genişleyen pencereye karşı (pencere − genişleyen)")
for w in [3, 5, 10]:
    sw = [mf(W[(w, y)]) for y in ALL_YEARS]
    print(f"  rolling-{w}: ort {np.mean(sw):.4f} | {cmp(ser['S'], sw)}")

print("\n[Kayma dedektörleri]")
if isinstance(K, dict) and "S" in K:
    for arm in "SBL":
        k = K[arm]
        print(f"  {arm}: hata {k['err_first']:.3f}→{k['err_last']:.3f}, on yılda {k['slope_per_decade']*100:+.2f} puan, doğrusal R²={k['linear_r2']:.2f}")
        print(f"     alarmlar: {k['alarms']}")
        print(f"     ≥2 dedektörün aynı yıl alarm verdiği yıllar: {k['years_two_or_more_agree']}")
        print(f"     3 permütasyonda aynı kalan: {k['perm_stable_of_3']}")
        print(f"     ADWIN delta taraması: {k['adwin_delta_sweep']}")
else:
    print("  ", K)

print(f"\n✅ Bitti. Çıktının TAMAMINI gönder. Kayıtlar: {CK}")
