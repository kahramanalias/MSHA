# =============================================================================
# What Can Be Known When a Mining Injury Is Reported? Temporal Availability and
# Severity Prediction in 24 Years of MSHA Records (Applied Sciences, 2026)
# Author: Abdulgani Kahraman, Balikesir University
#
# Step 6. Notification set over 2006-2023 (Tables 4 and 8, Figures 2 and 4), four learners on the
#   notification set (Table 5), hyperparameter sensitivity (Table S5), label maturation (Table 10),
#   1000-term vocabulary audit (Table S6, File S1) and event-only models (Table 6), excluded
#   records (Table S1), alerts per mine (Table S8), alert budgets (Table 7), PR curves, calibration
#   and decision curves (Table S7, Figure 3). Prints a JSON block used by step 8. Requires 2, 4, 5.
#
# Run order: 01 -> 02 -> (03) -> 04 -> 05 -> 06 -> 07 -> 08. See README.md.
# Written for Google Colab (each script was run as one cell). Outside Colab the
# scripts use the current folder as BASE; put Accidents.txt there.
# Comments and console messages are partly in Turkish; the code is unchanged
# from the runs reported in the paper except for the data-path lines marked
# "[repo]".
# =============================================================================

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MSHA — HAKEM REVİZYONU ANALİZLERİ (Applied Sciences, R1)   [v24-revision] ║
# ║  v21, v19 ve v22 kayıtlarını kullanır (Drive'da duruyor olmalı).          ║
# ║   N  yalnız yapısal alanlar, TAM DÖNEM 2006-2023 (birincil analiz)        ║
# ║   G  etiket olgunlaşma boşluğu: eğitim < y-1 ve < y-2                     ║
# ║   L  dört öğrenici, yalnız yapısal alanlar (2018-2023)                    ║
# ║   H  hiperparametre duyarlılığı (8 ayar, seçim önceki yıldan)             ║
# ║   D  eğitim yalnız belgelenmiş işe-dönüş/kapanış tarihi olan kayıtlarla   ║
# ║   V  1000 terimlik sözlüğün dökümü + olay-dışı terimler çıkarılmış model  ║
# ║   R  dışlanan kayıtların kategori dökümü + maden bazında uyarı oranı      ║
# ║   E  PR eğrileri (belirsizlikle), uyarı bütçeleri, kalibrasyon, karar     ║
# ║      eğrisi (hepsi önceki yılın verisiyle ayarlanır)                      ║
# ║  Süre: ~3,5 saat. Kopsa aynı hücreyi yeniden çalıştır; kaldığı yerden       ║
# ║  devam eder. En sonda ÖZET basılır: çıktının TAMAMINI gönder.             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import os, re, json, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
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
V21 = os.path.join(V7, "pastvocab_v21", "ckpt")
V19 = os.path.join(V7, "sensitivity_v19", "ckpt")
V22 = os.path.join(V7, "reviewer_v22", "ckpt")
OUT = os.path.join(V7, "revision_v24"); CK = os.path.join(OUT, "ckpt"); os.makedirs(CK, exist_ok=True)
SEED, SEVERE, MAX_TRAIN, N_TFIDF = 20260820, 3, 60_000, 100
ALL_YEARS = list(range(2006, 2024)); RECENT = list(range(2018, 2024)); VALY = list(range(2017, 2024))
print("[v24-revision] →", OUT)

LGB = dict(objective="multiclass", num_class=4, n_estimators=300, learning_rate=0.05,
           num_leaves=63, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
           is_unbalance=True, reg_alpha=0.1, reg_lambda=0.1, random_state=SEED,
           n_jobs=-1, verbose=-1)
# v21 ana analizinin sonuç-terimi listesi (100 terimlik sözlük)
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
# v19 / v22: 1000 terimlik sözlüğün sonuç-terimi listesi
OUTCOME_STEMS_V19 = ["fatal", "died", "death", "deceas", "kill", "dead", "doa", "hospital",
    "admitted", "icu", "ambulanc", "airlift", "life flight", "lifeflight", "medevac", "amput",
    "dismember", "paralys", "paraly", "quadripleg", "parapleg", "permanent", "disabilit",
    "disabl", "days away", "days lost", "lost time", "losttime", "restricted duty",
    "restricted work", "light duty", "lt duty", "time off", "off work", "returned to work",
    "return to work", "rtw", "first aid", "firstaid", "no lost", "released", "no treatment",
    "coroner", "autops", "funeral", "morgue"]
RE19 = re.compile("|".join(re.escape(s) for s in OUTCOME_STEMS_V19))

# ── olay-dışı (yaralanma / vücut bölgesi / tedavi / belirti) terim sözlüğü ──
LEX_PREFIX = {
    "nature": ("fractur", "lacerat", "contus", "dislocat", "concuss", "punctur", "abras", "ruptur",
               "sprain", "strain", "bruis", "tendon", "ligament", "hernia", "blister", "infect"),
    "treatment": ("surg", "sutur", "stitch", "diagnos", "prescri", "medicat", "physician", "therap"),
}
LEX_EXACT = {
    "nature": {"cut", "cuts", "gash", "wound", "wounds", "burn", "burns", "burned", "burnt", "crushed",
               "smashed", "mashed", "broke", "broken", "torn", "tore", "tear", "severed", "bleeding",
               "blood", "swelling", "swollen", "unconscious", "whiplash", "muscle", "nerve", "bone",
               "bones", "tissue", "skin", "nail"},
    "treatment": {"doctor", "doctors", "dr", "clinic", "medical", "treated", "treatment", "treat",
                  "xray", "xrays", "ray", "rays", "mri", "examined", "exam", "evaluated", "evaluation",
                  "splint", "cast", "er", "emergency", "nurse", "glued", "bandage", "staples", "taken"},
    "body": {"finger", "fingers", "thumb", "thumbs", "hand", "hands", "wrist", "wrists", "arm", "arms",
             "forearm", "elbow", "elbows", "shoulder", "shoulders", "back", "neck", "head", "face",
             "eye", "eyes", "ear", "ears", "nose", "mouth", "lip", "tooth", "teeth", "jaw", "chin",
             "forehead", "scalp", "chest", "rib", "ribs", "abdomen", "stomach", "hip", "hips", "groin",
             "leg", "legs", "knee", "knees", "ankle", "ankles", "foot", "feet", "toe", "toes", "heel",
             "shin", "thigh", "calf", "palm", "knuckle", "skull", "spine", "body", "index", "ring",
             "pinky", "fingertip"},
    "symptom": {"pain", "pains", "painful", "sore", "soreness", "hurt", "hurting", "hurts", "felt",
                "pop", "popped", "numb", "numbness", "dizzy", "ache", "aching", "discomfort",
                "tingling", "stiff", "stiffness"},
}
def lex_cat(term):
    toks = term.split()
    for cat in ("nature", "treatment", "body", "symptom"):
        ex = LEX_EXACT.get(cat, set()); pf = LEX_PREFIX.get(cat, ())
        if any((t in ex) or any(t.startswith(p) for p in pf) for t in toks): return cat
    return "event"

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
    print(f"  {name:40s} ({time.perf_counter()-t0:.0f}s)")
    return r

dS, featsS, _ = pload(os.path.join(V7, "arm_S", "ckpt", "data.pkl"))
print(f"  kayıt {len(dS):,} | yapısal alan {len(featsS)}")

def strat(d, n):
    if len(d) <= n: return d
    return d.groupby("TARGET", group_keys=False).apply(
        lambda g: g.sample(max(1, int(round(n * len(g) / len(d)))), random_state=SEED))

def vocab_before(t):                               # 100 terim; yalnız YEAR < t kayıtlarından (v21 ile özdeş)
    p = os.path.join(V21, f"vocab_{t}.pkl")
    if os.path.exists(p): return pload(p)["clean"]
    def make():
        cv = CountVectorizer(ngram_range=(1, 2), min_df=20, max_features=2000, stop_words="english", binary=True)
        M = cv.fit_transform(dS.loc[dS.YEAR < t, "_TXT"])
        nm = np.array(cv.get_feature_names_out()); dfq = np.asarray(M.sum(0)).ravel()
        return [x for x in nm[np.argsort(-dfq)] if not OUT_RE.search(x)][:N_TFIDF]
    return ck(f"vocab100_{t}", make)

def rank2000(t):                                   # v21 aday listesi (sık → seyrek), YEAR < t
    def make():
        cv = CountVectorizer(ngram_range=(1, 2), min_df=20, max_features=2000, stop_words="english", binary=True)
        M = cv.fit_transform(dS.loc[dS.YEAR < t, "_TXT"])
        nm = np.array(cv.get_feature_names_out()); dfq = np.asarray(M.sum(0)).ravel()
        return [str(x) for x in nm[np.argsort(-dfq)]]
    return ck(f"rank2000_{t}", make)
def vocab100_event(t): return [x for x in rank2000(t) if not OUT_RE.search(x) and lex_cat(x) == "event"][:N_TFIDF]

def rank1000(y):                                   # v19/v22 ile özdeş sıralama; eğitim örnekleminden
    def make():
        tr = strat(dS[dS.YEAR < y], MAX_TRAIN)
        cv = CountVectorizer(ngram_range=(1, 2), min_df=5, max_features=6000, binary=True)
        X = cv.fit_transform(tr["_TXT"].fillna("")); df = np.asarray(X.sum(0)).ravel()
        names = cv.get_feature_names_out(); order = np.argsort(-df)
        return [(str(names[i]), int(df[i])) for i in order]
    return ck(f"rank1000_{y}", make)
def vocab1000(y):       return [t for t, _ in rank1000(y) if not RE19.search(t)][:1000]
def vocab1000_event(y): return [t for t, _ in rank1000(y) if not RE19.search(t) and lex_cat(t) == "event"][:1000]

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

def pack(te, proba):
    yt = te["TARGET"].values.astype(np.int8); yp = np.argmax(proba, 1).astype(np.int8)
    return {"yt": yt, "yp": yp, "proba": proba, "p3": proba[:, 3]}
def lgbm(tr, te, vocab, **kw):
    Xtr, (Xte,) = mats(tr, [te], featsS, vocab)
    return lgb.LGBMClassifier(**{**LGB, **kw}).fit(Xtr, tr["TARGET"].values).predict_proba(Xte).astype(np.float32)
def run(y, vocab, **kw):
    te = dS[dS.YEAR == y]
    return pack(te, lgbm(strat(dS[dS.YEAR < y], MAX_TRAIN), te, vocab, **kw))

# ── mevcut kayıtlar ────────────────────────────────────────────────────────
A_S = {y: pload(os.path.join(V21, f"A_S_{y}.pkl")) for y in ALL_YEARS}        # yapısal + anlatı (100 terim)
A_B = {y: pload(os.path.join(V21, f"A_B_{y}.pkl")) for y in ALL_YEARS}        # değerlendirme seti
A_L = {y: pload(os.path.join(V21, f"A_L_{y}.pkl")) for y in ALL_YEARS}        # sonuç bilgili
T1000 = {y: pload(os.path.join(V19, f"S_C_tfidf1000_{y}.pkl")) for y in RECENT}

# ── N: yalnız yapısal alanlar, 2006-2023 ───────────────────────────────────
print("\n=== N: yalnız yapısal alanlar (2006-2017 yeni; 2018-2023 v21) ===")
NN = {}
for y in ALL_YEARS:
    p = os.path.join(V21, f"N_S_nonarr_{y}.pkl")
    NN[y] = pload(p) if os.path.exists(p) else ck(f"N_{y}", lambda y=y: run(y, []))

# ── G: etiket olgunlaşma boşluğu ───────────────────────────────────────────
print("\n=== G: boşluklu eğitim (M_t: YEAR < t; t+1 ve t+2 yıllarını tahmin eder) ===")
G = {}
for arm in ["N", "S"]:
    for t in range(2004, 2023):
        def f(arm=arm, t=t):
            tr = strat(dS[dS.YEAR < t], MAX_TRAIN)
            ys = [y for y in (t + 1, t + 2) if 2006 <= y <= 2023]
            tes = [dS[dS.YEAR == y] for y in ys]
            Xtr, Xs = mats(tr, tes, featsS, [] if arm == "N" else vocab_before(t))
            m = lgb.LGBMClassifier(**LGB).fit(Xtr, tr["TARGET"].values)
            return {y: pack(te, m.predict_proba(X).astype(np.float32)) for y, te, X in zip(ys, tes, Xs)}
        G[(arm, t)] = ck(f"G_{arm}_{t}", f)
GAP = {(arm, g): {y: G[(arm, y - g)][y] for y in ALL_YEARS} for arm in ["N", "S"] for g in (1, 2)}

# ── D: eğitim yalnız "etiketi o tarihte kesinleşmiş" kayıtlarla ─────────────
print("\n=== D: belgelenmiş işe-dönüş/kapanış tarihi test yılından önce olan kayıtlarla eğitim ===")
dL, featsL, _ = pload(os.path.join(V7, "arm_L", "ckpt", "data.pkl"))
assert dL.index.equals(dS.index)
RTW = pd.to_datetime(dL["RETURN_TO_WORK_DT"], format="%m/%d/%Y", errors="coerce")
if RTW.isna().mean() > 0.5: RTW = pd.to_datetime(dL["RETURN_TO_WORK_DT"], errors="coerce")
def mature(y):                                     # tarih < 1 Ocak y; tarih yoksa en az iki tam yıl geçmiş olmalı
    cut = pd.Timestamp(year=y, month=1, day=1)
    return (dS.YEAR < y) & ((RTW < cut) | (RTW.isna() & (dS.YEAR < y - 2)))
DD = {}
for arm in ["N", "S"]:
    for y in ALL_YEARS:
        def f(arm=arm, y=y):
            te = dS[dS.YEAR == y]
            return pack(te, lgbm(strat(dS[mature(y)], MAX_TRAIN), te, [] if arm == "N" else vocab_before(y)))
        DD[(arm, y)] = ck(f"D_{arm}_{y}", f)

# ── T5N: yalnız yapısal alanlar için duyarlılık varyantları ────────────────
print("\n=== T5N: yalnız yapısal alanlar, dengeli ağırlık ve one-hot lojistik (2018-2023) ===")
T5N = {}
for y in RECENT:
    T5N[("balanced", y)] = ck(f"T5N_balanced_{y}", lambda y=y: run(y, [], class_weight="balanced", is_unbalance=False))
    def fE(y=y):
        tr, te = strat(dS[dS.YEAR < y], MAX_TRAIN), dS[dS.YEAR == y]
        num = tr[featsS].select_dtypes(include=[np.number]).columns.tolist(); cat = [c for c in featsS if c not in num]
        oh = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20).fit(tr[cat].astype(str).fillna("NA"))
        def T(dd):
            parts = [oh.transform(dd[cat].astype(str).fillna("NA"))]
            if num: parts.append(sp.csr_matrix(dd[num].apply(pd.to_numeric, errors="coerce").fillna(-1).to_numpy(np.float32)))
            return sp.hstack(parts).tocsr()
        sc = MaxAbsScaler().fit(T(tr))
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(T(tr)), tr["TARGET"].values)
        return pack(te, m.predict_proba(sc.transform(T(te))).astype(np.float32))
    T5N[("logreg_onehot", y)] = ck(f"T5N_logregoh_{y}", fE)

# ── L: dört öğrenici, yalnız yapısal alanlar ───────────────────────────────
print("\n=== L: öğreniciler, yalnız yapısal alanlar (2018-2023) ===")
def make(name):
    if name == "RF": return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                                   class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    if name == "HistGB": return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
                                                               min_samples_leaf=30, l2_regularization=0.1,
                                                               class_weight="balanced", random_state=SEED)
    return LogisticRegression(max_iter=2000, class_weight="balanced")
T4N = {}
for name in ["RF", "HistGB", "LogReg"]:
    for y in RECENT:
        def f(name=name, y=y):
            tr, te = strat(dS[dS.YEAR < y], MAX_TRAIN), dS[dS.YEAR == y]
            Xtr, (Xte,) = mats(tr, [te], featsS, [])
            if name == "HistGB": Xtr, Xte = Xtr.toarray(), Xte.toarray()
            m = make(name).fit(Xtr, tr["TARGET"].values)
            return {"yt": te["TARGET"].values.astype(np.int8), "yp": m.predict(Xte).astype(np.int8)}
        T4N[(name, y)] = ck(f"T4N_{name}_{y}", f)

# ── H: hiperparametre duyarlılığı ──────────────────────────────────────────
print("\n=== H: hiperparametre duyarlılığı (2017-2023; seçim önceki yıldan) ===")
HP = {"c0_default": {},
      "c1_leaves15": dict(num_leaves=15), "c2_leaves255": dict(num_leaves=255),
      "c3_600trees_lr0.025": dict(n_estimators=600, learning_rate=0.025),
      "c4_150trees_lr0.1": dict(n_estimators=150, learning_rate=0.1),
      "c5_minchild100": dict(min_child_samples=100),
      "c6_rowbag0.8_col0.5": dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.5),
      "c7_reg1.0": dict(reg_alpha=1.0, reg_lambda=1.0)}
H = {}
for arm in ["N", "S"]:
    for c, kw in HP.items():
        for y in VALY:
            if c == "c0_default":
                H[(arm, c, y)] = NN[y] if arm == "N" else A_S[y]; continue
            H[(arm, c, y)] = ck(f"H_{arm}_{c}_{y}", lambda arm=arm, kw=kw, y=y:
                                run(y, [] if arm == "N" else vocab_before(y), **kw))

# ── V: 1000 terimlik sözlük dökümü ve olay-dışı terimler çıkarılmış model ───
print("\n=== V: 1000 terimlik sözlük ===")
T1000[2017] = ck("T1000_2017", lambda: run(2017, vocab1000(2017)))
T1000E = {y: ck(f"T1000E_{y}", lambda y=y: run(y, vocab1000_event(y))) for y in VALY}
T100E = {y: ck(f"T100E_{y}", lambda y=y: run(y, vocab100_event(y))) for y in VALY}
vrows = {}
for y in RECENT:
    n_kept = 0
    for t, df in rank1000(y):                      # sık → seyrek; ilk 1000 taranmamış terim tutulur
        scr = bool(RE19.search(t))
        r = vrows.setdefault(t, {"term": t, "screen": "removed" if scr else "kept", "category": lex_cat(t), "years": set(), "df2023": None})
        r["years"].add(y)
        if y == 2023: r["df2023"] = df
        if not scr:
            n_kept += 1
            if n_kept == 1000: break
vdf = pd.DataFrame([{**r, "years": "".join("1" if y in r["years"] else "0" for y in RECENT)} for r in vrows.values()])
vdf = vdf.sort_values(["screen", "category", "term"]).reset_index(drop=True)
vdf.to_csv(os.path.join(OUT, "vocabulary_1000_terms.csv"), index=False)

# ── R: ham veri — dışlanan kayıtlar ve maden kimliği ───────────────────────
print("\n=== R: ham veri ===")
SEVERITY_MAP = {"ALL OTHER CASES (INCL 1ST AID)": 0, "NO DYS AWY FRM WRK,NO RSTR ACT": 0,
                "DAYS RESTRICTED ACTIVITY ONLY": 1, "DAYS AWAY FROM WORK ONLY": 2,
                "DYS AWY FRM WRK & RESTRCTD ACT": 2, "PERM TOT OR PERM PRTL DISABLTY": 3, "FATALITY": 3}
RAW = None
try:
    src = next(p for p in [os.path.join(BASE, "MSHA_pipeline", "ckpt", "s1_data.pkl"),
                           os.path.join(BASE, "Accidents.txt"),
                           os.path.join(BASE, "NIOSHA veri raporlar", "Accidents.txt")] if os.path.exists(p))
    if src.endswith(".pkl"):
        obj = pload(src); raw = (obj[0] if isinstance(obj, (tuple, list)) else obj).copy()
    else:
        raw = pd.read_csv(src, sep="|", encoding="latin-1", on_bad_lines="skip", low_memory=False)
    raw.columns = [c.strip().upper() for c in raw.columns]
    DTC = next(c for c in ["ACCIDENT_DT", "ACCIDENT_DATE"] if c in raw.columns)
    raw["_dt"] = pd.to_datetime(raw[DTC], errors="coerce"); raw["YEAR"] = raw["_dt"].dt.year
    raw = raw.dropna(subset=["YEAR", "DEGREE_INJURY"]); raw["YEAR"] = raw["YEAR"].astype(int)
    raw = raw[raw["YEAR"].between(2000, 2023)].sort_values("_dt").reset_index(drop=True)
    raw["_deg"] = raw["DEGREE_INJURY"].astype(str).str.strip().str.upper()
    raw["_lvl"] = raw["_deg"].map({k.upper(): v for k, v in SEVERITY_MAP.items()})
    RAW = raw
    print(f"  kaynak: {src} | {len(raw):,} satır")
except Exception as e:
    print("  ⚠ ham veri okunamadı; R adımı atlandı:", repr(e)[:200])

# ══ yardımcılar ════════════════════════════════════════════════════════════
def mf(r): return f1_score(r["yt"], r["yp"], average="macro", zero_division=0)
def cat_(recs, k): return np.concatenate([r[k] for r in recs])
def sev(yt, flag):
    tp = int((flag & (yt == SEVERE)).sum()); fl = int(flag.sum()); ns = int((yt == SEVERE).sum())
    P = tp / fl if fl else 0.0; R = tp / ns if ns else 0.0
    return dict(P=P, R=R, F1=(2*P*R/(P+R) if P+R else 0.0), flagged=fl, tp=tp, missed=ns-tp,
                NNA=(fl/tp if tp else float("nan")), n=len(yt), ns=ns)
def ap(recs):
    if any(r.get("p3") is None for r in recs): return float("nan")
    return float(average_precision_score(cat_(recs, "yt") == SEVERE, cat_(recs, "p3")))
def line(label, recs):
    s = sev(cat_(recs, "yt"), cat_(recs, "yp") == SEVERE)
    return (f"  {label:50s} macroF1={np.mean([mf(r) for r in recs]):.3f}  P={s['P']:.3f} R={s['R']:.3f} F1={s['F1']:.3f} "
            f"flag={s['flagged']} missed={s['missed']} NNA={s['NNA']:.2f} AP={ap(recs):.3f}")
rng = np.random.default_rng(SEED)
def cmp(a, b):
    dd = np.asarray(b) - np.asarray(a)
    bs = np.array([rng.choice(dd, len(dd), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    try: _, p = stats.wilcoxon(dd)
    except Exception: p = float("nan")
    return f"Δ={dd.mean():+.4f} 95%CI [{lo:+.4f}, {hi:+.4f}] Wilcoxon p={p:.1e}"
def best_thr(yt, p3):
    pr, rc, th = precision_recall_curve(yt == SEVERE, p3)
    f = 2 * pr * rc / np.maximum(pr + rc, 1e-12); i = int(np.nanargmax(f[:-1])); return float(th[i])
def with_prev_thr(D):                      # D: {y: rec} 2017-2023; eşik y-1 tahminlerinden
    out = []
    for y in RECENT:
        thr = best_thr(D[y - 1]["yt"], D[y - 1]["p3"]); pr = D[y]["proba"]
        yp = np.argmax(pr[:, :3], 1); yp[pr[:, 3] >= thr] = SEVERE
        out.append({"yt": D[y]["yt"], "yp": yp.astype(np.int8), "p3": D[y]["p3"], "thr": thr})
    return out
def table3(recs, label):
    yt, yp = cat_(recs, "yt"), cat_(recs, "yp")
    P, R, F, Sp = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2, 3], zero_division=0)
    print(f"  [{label}] havuzlanmış")
    for k in range(4): print(f"    L{k}: P={P[k]:.3f} R={R[k]:.3f} F1={F[k]:.3f} n={Sp[k]}")
    print(f"    macro: P={P.mean():.3f} R={R.mean():.3f} F1={F.mean():.3f} n={Sp.sum()} | acc={np.mean(yt == yp):.3f}")
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3]); print("    confusion (row=true):", cm.tolist())
    return cm.tolist()
JS = {}

print("\n" + "█" * 100 + "\n  ÖZET v24\n" + "█" * 100)

# ── [1] dört senaryo, 2006-2023 ────────────────────────────────────────────
print("\n[1] Senaryolar 2006-2023 (yıllık ortalamalar)")
SETS = [("N", "Notification (structured only)", NN), ("S", "Structured + narrative", A_S),
        ("B", "Assessment", A_B), ("L", "Outcome-informed", A_L)]
ser = {}
for k, n, D in SETS:
    recs = [D[y] for y in ALL_YEARS]; ser[k] = [mf(r) for r in recs]
    ba = np.mean([balanced_accuracy_score(r["yt"], r["yp"]) for r in recs])
    sr = np.mean([((r["yp"] == SEVERE) & (r["yt"] == SEVERE)).sum() / max((r["yt"] == SEVERE).sum(), 1) for r in recs])
    s = sev(cat_(recs, "yt"), cat_(recs, "yp") == SEVERE)
    print(f"  {n:32s} macroF1={np.mean(ser[k]):.4f} SD={np.std(ser[k], ddof=1):.4f} balAcc={ba:.4f} "
          f"sevRecall(annual mean)={sr:.4f} | pooled L3: P={s['P']:.3f} R={s['R']:.3f} F1={s['F1']:.3f} tp={s['tp']}/{s['ns']} AP={ap(recs):.3f}")
for a, b in [("N", "S"), ("N", "B"), ("N", "L"), ("S", "B"), ("S", "L")]:
    print(f"  {b} − {a}: {cmp(ser[a], ser[b])}")
for k in "NSBL": print(f"  annual {k}: " + ", ".join(f"{v:.4f}" for v in ser[k]))
JS["annual"] = {k: [round(float(v), 4) for v in ser[k]] for k in "NSBL"}
JS["cm_N"] = table3([NN[y] for y in ALL_YEARS], "Notification, 2006-2023")
JS["cm_S"] = table3([A_S[y] for y in ALL_YEARS], "Structured + narrative, 2006-2023")
print("  2018-2023:")
for k, n, D in SETS[:3]: print(line(n, [D[y] for y in RECENT]))

# ── [2] öğreniciler, yalnız yapısal ────────────────────────────────────────
print("\n[2] Öğreniciler, yalnız yapısal alanlar, 2018-2023")
print(line("Notification LightGBM", [NN[y] for y in RECENT]))
for nm in ["RF", "HistGB", "LogReg"]:
    recs = [T4N[(nm, y)] for y in RECENT]; s = sev(cat_(recs, "yt"), cat_(recs, "yp") == SEVERE)
    print(f"  {'Notification ' + nm:50s} macroF1={np.mean([mf(r) for r in recs]):.3f}  P={s['P']:.3f} R={s['R']:.3f} F1={s['F1']:.3f} "
          f"flag={s['flagged']} missed={s['missed']} NNA={s['NNA']:.2f}")

print(line("Notification LightGBM balanced weights", [T5N[("balanced", y)] for y in RECENT]))
print(line("Notification LogReg one-hot", [T5N[("logreg_onehot", y)] for y in RECENT]))
def oracle(label, recs):
    yt = cat_(recs, "yt") == SEVERE; p3 = cat_(recs, "p3")
    pr, rc, th = precision_recall_curve(yt, p3); f = 2 * pr * rc / np.maximum(pr + rc, 1e-12); i = int(np.nanargmax(f[:-1]))
    fl = int((p3 >= th[i]).sum()); tp = int(((p3 >= th[i]) & yt).sum())
    print(f"  {label:50s} TEST-SET thr (optimistic): F1={f[i]:.3f} P={pr[i]:.3f} R={rc[i]:.3f} flag={fl} missed={int(yt.sum()) - tp} NNA={fl / max(tp, 1):.2f}")
oracle("Notification", [NN[y] for y in RECENT]); oracle("Structured + 100-term", [A_S[y] for y in RECENT])
oracle("Structured + 1000-term", [T1000[y] for y in RECENT]); oracle("Assessment", [A_B[y] for y in RECENT])
print(line("Notification + prior-year thr", with_prev_thr(NN)))

# ── [3] etiket olgunlaşma boşluğu ──────────────────────────────────────────
print("\n[3] Etiket olgunlaşma boşluğu (eğitim: YEAR < y − gap)")
for arm, n, D in [("N", "Notification", NN), ("S", "Structured + narrative", A_S)]:
    base = [mf(D[y]) for y in ALL_YEARS]
    print(f"  {n} gap0 (main): 2006-2023 macroF1={np.mean(base):.4f} AP={ap([D[y] for y in ALL_YEARS]):.3f}")
    print(line(f"{n} gap0 2018-2023", [D[y] for y in RECENT]))
    for g in (1, 2):
        s_ = [mf(GAP[(arm, g)][y]) for y in ALL_YEARS]
        print(f"  {n} gap{g}: 2006-2023 macroF1={np.mean(s_):.4f} AP={ap([GAP[(arm, g)][y] for y in ALL_YEARS]):.3f} | gap{g} − main: {cmp(base, s_)}")
        print(line(f"{n} gap{g} 2018-2023", [GAP[(arm, g)][y] for y in RECENT]))
        print(line(f"{n} gap{g} 2006-2023 pooled", [GAP[(arm, g)][y] for y in ALL_YEARS]))
    print(line(f"{n} gap0 2006-2023 pooled", [D[y] for y in ALL_YEARS]))

print("  Belgelenmiş kapanış tarihiyle eğitim (D):")
for y in (2006, 2012, 2018, 2023):
    prevy = (dS.YEAR == y - 1); m = mature(y)
    print(f"    test {y}: eğitim kayıtlarının %{(1 - m.sum() / (dS.YEAR < y).sum()) * 100:.1f}'i çıkarıldı; "
          f"y−1 yılının %{(1 - (m & prevy).sum() / prevy.sum()) * 100:.1f}'i | çıkarılanların sınıf dağılımı "
          f"{dS.loc[(dS.YEAR < y) & ~m, 'TARGET'].value_counts(normalize=True).round(3).sort_index().to_dict()}")
print(f"    işe-dönüş tarihi eksik oranı: tümü {RTW.isna().mean():.3f} | sınıfa göre "
      f"{RTW.isna().groupby(dS.TARGET).mean().round(3).to_dict()}")
for arm, n, D in [("N", "Notification", NN), ("S", "Structured + narrative", A_S)]:
    base = [mf(D[y]) for y in ALL_YEARS]; s_ = [mf(DD[(arm, y)]) for y in ALL_YEARS]
    print(f"  {n} documented-disposition: 2006-2023 macroF1={np.mean(s_):.4f} AP={ap([DD[(arm, y)] for y in ALL_YEARS]):.3f} | − main: {cmp(base, s_)}")
    print(line(f"{n} documented-disposition 2018-2023", [DD[(arm, y)] for y in RECENT]))
    print(line(f"{n} documented-disposition 2006-2023 pooled", [DD[(arm, y)] for y in ALL_YEARS]))

# ── [4] hiperparametre duyarlılığı ─────────────────────────────────────────
print("\n[4] Hiperparametre duyarlılığı, 2018-2023")
for arm, n in [("N", "Notification"), ("S", "Structured + narrative")]:
    for c in HP: print(line(f"{n} {c}", [H[(arm, c, y)] for y in RECENT]))
    for crit, fn in [("macroF1", mf), ("AP", lambda r: average_precision_score(r["yt"] == SEVERE, r["p3"]))]:
        pick = {y: max(HP, key=lambda c: fn(H[(arm, c, y - 1)])) for y in RECENT}
        print(line(f"{n} tuned (prior-year {crit})", [H[(arm, pick[y], y)] for y in RECENT]) + f"  picks={[pick[y][:2] for y in RECENT]}")
        Dt = {y: H[(arm, pick[y], y)] for y in RECENT}
        outs = []
        for y in RECENT:
            pv = H[(arm, pick[y], y - 1)]; thr = best_thr(pv["yt"], pv["p3"]); pr = Dt[y]["proba"]
            yp = np.argmax(pr[:, :3], 1); yp[pr[:, 3] >= thr] = SEVERE
            outs.append({"yt": Dt[y]["yt"], "yp": yp.astype(np.int8), "p3": Dt[y]["p3"]})
        print(line(f"{n} tuned ({crit}) + prior-year thr", outs))
    rr = [(np.mean([mf(H[(arm, c, y)]) for y in RECENT]), ap([H[(arm, c, y)] for y in RECENT])) for c in HP]
    print(f"  {n}: macroF1 aralığı {min(r[0] for r in rr):.3f}-{max(r[0] for r in rr):.3f} | AP aralığı {min(r[1] for r in rr):.3f}-{max(r[1] for r in rr):.3f}")

# ── [5] sözlük dökümü ──────────────────────────────────────────────────────
print("\n[5] 1000 terimlik sözlük (değerlendirme yılları 2018-2023)")
for y in RECENT:
    v = vocab1000(y); c = pd.Series([lex_cat(t) for t in v]).value_counts().to_dict()
    v100 = vocab_before(y); c100 = pd.Series([lex_cat(t) for t in v100]).value_counts().to_dict()
    nrem = sum(1 for t in vrows.values() if y in t["years"] and t["screen"] == "removed")
    print(f"  {y}: 1000 terim → {c} | taramayla çıkarılan aday {nrem} | 100 terim → {c100}")
kept_u = vdf[vdf.screen == "kept"]
print(f"  birleşim: tutulan {len(kept_u)} terim, taramayla çıkarılan {int((vdf.screen == 'removed').sum())}; "
      f"altı yılın hepsinde olan {int((kept_u.years == '111111').sum())}")
print("  kategori (birleşim, tutulan):", kept_u.category.value_counts().to_dict())
print(line("Structured + 1000-term (v19)", [T1000[y] for y in RECENT]))
print(line("Structured + 1000-term + prior-year thr", with_prev_thr(T1000)))
print(line("Structured + 1000 event-only terms", [T1000E[y] for y in RECENT]))
print(line("Structured + 1000 event-only + prior-year thr", with_prev_thr(T1000E)))
print(line("Structured + 100-term (main)", [A_S[y] for y in RECENT]))
print(line("Structured + 100 event-only terms", [T100E[y] for y in RECENT]))
print(line("Structured + 100 event-only + prior-year thr", with_prev_thr(T100E)))
print("  100 terimlik sözlük, olay-dışı terimler (2023):", [t for t in vocab_before(2023) if lex_cat(t) != "event"])
print(line("Structured only", [NN[y] for y in RECENT]))
print("  ###VOCAB_BEGIN### term|screen|category|years(2018..2023)|df2023")
for r in vdf.itertuples(): print(f"  {r.term}|{r.screen}|{r.category}|{r.years}|{'' if pd.isna(r.df2023) else int(r.df2023)}")
print("  ###VOCAB_END###")

# ── [6] ham veri ───────────────────────────────────────────────────────────
print("\n[6] Dışlanan kayıtlar ve maden bazında uyarılar")
MINE = None
if RAW is not None:
    vc = RAW.groupby("_deg").size().sort_values(ascending=False)
    for k, v in vc.items():
        lv = RAW.loc[RAW._deg == k, "_lvl"].iloc[0]
        print(f"  {k:36s} {v:8,d}  {'EXCLUDED' if pd.isna(lv) else 'L' + str(int(lv))}")
    print(f"  toplam {len(RAW):,} | dahil {int(RAW._lvl.notna().sum()):,} | dışlanan {int(RAW._lvl.isna().sum()):,}")
    inc = RAW[RAW._lvl.notna()]
    TXT = next(c for c in ["NARRATIVE", "ACCIDENT_DESCRIPTION"] if c in inc.columns)
    itx = inc[TXT].fillna("").astype(str).str.lower().values
    ok = (len(inc) == len(dS)) and bool((inc["YEAR"].values == dS["YEAR"].values).all()) and \
         bool((inc["_lvl"].astype(int).values == dS["TARGET"].values).all()) and bool((itx == dS["_TXT"].values).all())
    print("  satır hizalaması (ham ↔ analiz verisi):", "TAMAM" if ok else "sıra farklı; anahtarla eşleştiriliyor")
    if "MINE_ID" in inc.columns:
        if ok:
            MINE = pd.Series(inc["MINE_ID"].astype(str).values, index=dS.index)
        else:                                      # (yıl, sınıf, anlatı) anahtarıyla; belirsiz anahtarlar "NA"
            key_i = pd.Series(list(zip(inc["YEAR"].values, inc["_lvl"].astype(int).values, itx)))
            key_d = pd.Series(list(zip(dS["YEAR"].values, dS["TARGET"].values, dS["_TXT"].values)))
            mp = pd.DataFrame({"k": key_i, "m": inc["MINE_ID"].astype(str).values}).drop_duplicates("k", keep=False).set_index("k")["m"]
            MINE = pd.Series(key_d.map(mp).fillna("NA").values, index=dS.index)
            print(f"  anahtarla eşleşen kayıt oranı: {(MINE != 'NA').mean():.3f}")
    if MINE is not None:
        m_recent = np.concatenate([MINE[dS.YEAR == y].values for y in RECENT])
        y_recent = np.concatenate([np.full(int((dS.YEAR == y).sum()), y) for y in RECENT])
        print(f"  2018-2023: kayıt {len(m_recent):,} | farklı maden {len(set(m_recent)):,} | kayıt bildiren maden-yıl {len(set(zip(m_recent, y_recent))):,}")

# ── [7] PR eğrileri, uyarı bütçeleri, kalibrasyon, karar eğrisi ────────────
print("\n[7] Ağır sınıf: PR eğrisi, uyarı bütçesi, kalibrasyon, karar eğrisi (2018-2023)")
MODELS = [("N", "Notification (structured only)", NN), ("S100", "Structured + 100-term narrative", A_S),
          ("S1000", "Structured + 1000-term narrative", T1000), ("S1000E", "Structured + 1000 event-only terms", T1000E),
          ("B", "Assessment", A_B)]
GRID = [round(x, 2) for x in np.arange(0.02, 1.0001, 0.02)]
def pr_grid(yt, p):
    pr, rc, _ = precision_recall_curve(yt, p); cm = np.maximum.accumulate(pr)
    return np.array([cm[max(int((rc >= r).sum()) - 1, 0)] for r in GRID])
NB_ = 1000; brng = np.random.default_rng(SEED + 7)
BOOT = [brng.choice(RECENT, len(RECENT), replace=True) for _ in range(NB_)]
JS["grid"] = GRID; JS["pr"] = {}; JS["ap"] = {}
for k, n, D in MODELS:
    yt = {y: D[y]["yt"] == SEVERE for y in RECENT}; p = {y: D[y]["p3"] for y in RECENT}
    full = pr_grid(np.concatenate([yt[y] for y in RECENT]), np.concatenate([p[y] for y in RECENT]))
    apf = average_precision_score(np.concatenate([yt[y] for y in RECENT]), np.concatenate([p[y] for y in RECENT]))
    bs, ba = [], []
    for ys in BOOT:
        a, b = np.concatenate([yt[y] for y in ys]), np.concatenate([p[y] for y in ys])
        bs.append(pr_grid(a, b)); ba.append(average_precision_score(a, b))
    lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0); alo, ahi = np.percentile(ba, [2.5, 97.5])
    JS["pr"][k] = {"p": [round(float(v), 4) for v in full], "lo": [round(float(v), 4) for v in lo], "hi": [round(float(v), 4) for v in hi]}
    JS["ap"][k] = [round(float(apf), 4), round(float(alo), 4), round(float(ahi), 4)]
    j25, j50, j75 = GRID.index(0.26), GRID.index(0.5), GRID.index(0.76)
    print(f"  {n:36s} AP={apf:.3f} [{alo:.3f}, {ahi:.3f}] | precision at recall≥0.50: {full[j50]:.3f} [{lo[j50]:.3f}, {hi[j50]:.3f}]"
          f" | ≥0.26: {full[j25]:.3f} | ≥0.76: {full[j75]:.3f}")
prev = np.mean(np.concatenate([NN[y]["yt"] for y in RECENT]) == SEVERE)
print(f"  prevalans (2018-2023) = {prev:.4f}")

def yearly_counts(flags, D):               # yıl bazında (tp, flagged, n_severe, n)
    return {y: (int((flags[y] & (D[y]["yt"] == SEVERE)).sum()), int(flags[y].sum()),
                int((D[y]["yt"] == SEVERE).sum()), len(flags[y])) for y in RECENT}
def boot_ci(cnt, fn):
    v = [fn(*np.sum([cnt[y] for y in ys], axis=0)) for ys in BOOT]; return np.nanpercentile(v, [2.5, 97.5])
BUD = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
print("\n  Uyarı bütçesi (eşik = önceki yılın tahminlerinde en yüksek %b; sonra y yılına uygulanır)")
FLAGS = {}
for k, n, D in MODELS[:4]:
    for b in BUD:
        fl = {y: D[y]["p3"] >= np.quantile(D[y - 1]["p3"], 1 - b) for y in RECENT}; FLAGS[(k, b)] = fl
        cnt = yearly_counts(fl, D); tp, f_, ns, N = np.sum([cnt[y] for y in RECENT], axis=0)
        rlo, rhi = boot_ci(cnt, lambda tp, f, ns, N: tp / ns)
        plo, phi = boot_ci(cnt, lambda tp, f, ns, N: tp / f if f else np.nan)
        print(f"  {n:36s} budget={b*100:4.1f}% → alerts={f_:5d} ({f_/N*100:4.1f}% of records, {f_/6:.0f}/yr) recall={tp/ns:.3f} [{rlo:.3f}, {rhi:.3f}] "
              f"precision={tp/max(f_,1):.3f} [{plo:.3f}, {phi:.3f}] NNA={f_/max(tp,1):.2f} missed={ns-tp} of {ns}")

def logit(p): p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def calib_stats(yt, p):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], yt.astype(int))
    q = np.quantile(p, np.linspace(0, 1, 11)); idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
    ece = sum((idx == i).mean() * abs(p[idx == i].mean() - yt[idx == i].mean()) for i in range(10) if (idx == i).any())
    return dict(mean_pred=float(p.mean()), prev=float(yt.mean()), brier=float(np.mean((p - yt) ** 2)),
                slope=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]), ece=float(ece))
EDGES = [0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0001]
PT = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]
print("\n  Kalibrasyon (Platt ölçekleme önceki yılın tahminlerinde kurulur) ve karar eğrisi")
JS["dca"] = {"pt": PT, "all": [round(float(prev - (1 - prev) * t / (1 - t)), 5) for t in PT]}; JS["cal"] = {}
PCAL = {}
for k, n, D in MODELS[:4]:
    pc = {}
    for y in RECENT:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(D[y - 1]["p3"])[:, None], D[y - 1]["yt"] == SEVERE)
        pc[y] = lr.predict_proba(logit(D[y]["p3"])[:, None])[:, 1]
    PCAL[k] = pc
    yt = np.concatenate([D[y]["yt"] == SEVERE for y in RECENT]).astype(float)
    raw_, cal_ = np.concatenate([D[y]["p3"] for y in RECENT]).astype(float), np.concatenate([pc[y] for y in RECENT])
    a, b = calib_stats(yt, raw_), calib_stats(yt, cal_)
    print(f"  {n}")
    for lab, s in [("raw", a), ("recalibrated", b)]:
        print(f"    {lab:13s} mean predicted={s['mean_pred']:.4f} observed={s['prev']:.4f} slope={s['slope']:.2f} "
              f"intercept={s['intercept']:+.2f} Brier={s['brier']:.5f} ECE={s['ece']:.4f}")
    bins = []
    for lo_, hi_ in zip(EDGES[:-1], EDGES[1:]):
        m = (cal_ >= lo_) & (cal_ < hi_)
        bins.append([lo_, min(hi_, 1.0), int(m.sum()), round(float(cal_[m].mean()), 4) if m.any() else None,
                     round(float(yt[m].mean()), 4) if m.any() else None])
    print("    reliability (recalibrated) [lo, hi, n, mean predicted, observed]:", bins)
    JS["cal"][k] = {"raw": {kk: round(v, 5) for kk, v in a.items()}, "cal": {kk: round(v, 5) for kk, v in b.items()}, "bins": bins}
    nb, nlo, nhi, rows = [], [], [], []
    for t in PT:
        fl = {y: pc[y] >= t for y in RECENT}; cnt = yearly_counts(fl, D)
        f_nb = lambda tp, f, ns, N, t=t: (tp - (f - tp) * t / (1 - t)) / N
        tp, f_, ns, N = np.sum([cnt[y] for y in RECENT], axis=0); lo_, hi_ = boot_ci(cnt, f_nb)
        nb.append(round(float(f_nb(tp, f_, ns, N)), 5)); nlo.append(round(float(lo_), 5)); nhi.append(round(float(hi_), 5))
        rows.append(f"pt={t:.3f} (FN:FP≈{(1-t)/t:.0f}:1) alerts={f_} tp={tp} missed={ns-tp} NNA={(f_/tp if tp else float('nan')):.2f} "
                    f"NB={nb[-1]:+.5f} [{nlo[-1]:+.5f}, {nhi[-1]:+.5f}] NB_all={prev - (1 - prev) * t / (1 - t):+.5f}")
    JS["dca"][k] = {"nb": nb, "lo": nlo, "hi": nhi}
    for r in rows: print("     ", r)

# ── maden bazında uyarı oranı ──────────────────────────────────────────────
if MINE is not None:
    print("\n  Maden ve zaman bazında uyarı oranı (2018-2023)")
    def v21p(name, y): return pload(os.path.join(V21, f"{name}_{y}.pkl"))
    def v22p(name, y): return pload(os.path.join(V22, f"{name}_{y}.pkl"))
    CFG = {"Structured + 100-term, argmax": {y: A_S[y]["yp"] == SEVERE for y in RECENT}}
    try: CFG["Structured + 100-term, prior-year F1 thr"] = {y: v21p("T5_S_prevthr", y)["yp"] == SEVERE for y in RECENT}
    except Exception: pass
    try:
        CFG["Structured + 1000-term, prior-year F1 thr"] = {y: v22p("C_S_t1000_prevthr", y)["yp"] == SEVERE for y in RECENT}
        CFG["Notification, prior-year F1 thr"] = {y: v22p("N_S_nonarr_prevthr", y)["yp"] == SEVERE for y in RECENT}
    except Exception: pass
    for k_, b_ in [("N", 0.02), ("N", 0.05), ("S100", 0.02), ("S1000", 0.02)]:
        CFG[f"{k_} budget {b_*100:.0f}%"] = FLAGS[(k_, b_)]
    tot_my = len(set(zip(m_recent, y_recent)))
    for lab, fl in CFG.items():
        f = np.concatenate([fl[y] for y in RECENT]); my = pd.Series(list(zip(m_recent[f], y_recent[f])))
        per = my.value_counts() if len(my) else pd.Series(dtype=int)
        print(f"  {lab:44s} alerts={int(f.sum())} ({f.sum()/6:.0f}/yr, {f.sum()/312:.1f}/week) | alerted mine-years={len(per)} of {tot_my} "
              f"({len(per)/tot_my*100:.1f}%) | max alerts in one mine-year={int(per.max()) if len(per) else 0} | "
              f"mine-years with 1 alert={int((per == 1).sum())}")

print("\n###JSON_BEGIN###")
print(json.dumps(JS, separators=(",", ":")))
print("###JSON_END###")
print(f"\n✅ Bitti. Çıktının TAMAMINI gönder (çok uzunsa .txt olarak ekle). Kayıtlar: {CK}\n"
      f"   Sözlük dosyası: {os.path.join(OUT, 'vocabulary_1000_terms.csv')}")
