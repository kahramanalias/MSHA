# =============================================================================
# What Can Be Known When a Mining Injury Is Reported? Temporal Availability and
# Severity Prediction in 24 Years of MSHA Records (Applied Sciences, 2026)
# Author: Abdulgani Kahraman, Balikesir University
#
# Step 1. Reads the MSHA Accidents file, maps DEGREE_INJURY to the four severity levels (Table S1),
#   classifies every field by availability (Table S2) and writes one data set per feature set:
#   arm_S = notification/completed-report fields, arm_B = + assessment (clinical) fields,
#   arm_L = + post-closure and severity-triggered fields (outcome-informed).
#   Outputs: MSHA_AEJ_v7/arm_{S,B,L}/ckpt/data.pkl (used by all later steps) and first-pass results.
#
# Run order: 01 -> 02 -> (03) -> 04 -> 05 -> 06 -> 07 -> 08. See README.md.
# Written for Google Colab (each script was run as one cell). Outside Colab the
# scripts use the current folder as BASE; put Accidents.txt there.
# Comments and console messages are partly in Turkish; the code is unchanged
# from the runs reported in the paper except for the data-path lines marked
# "[repo]".
# =============================================================================

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MSHA — ÇOK KOLLU KARŞILAŞTIRMA  (AEJ)                                   ║
# ║  TEK HÜCRE, kopmaya dayanıklı. Aynı hücreyi tekrar çalıştır → devam.     ║
# ║                                                                          ║
# ║  Üç kol, TEK ve AYNI nedensel boru hattı:                                ║
# ║    S  strict      — yalnızca olay anında mevcut alanlar  (ZATEN KOŞTU)   ║
# ║    B  borderline  — + triyajda bilinen klinik alanlar                    ║
# ║    L  leaky       — + olay-sonrası alanlar (EAAI sürümünün özellik seti) ║
# ║                                                                          ║
# ║  Tez: raporlanan "kavram kayması" özellik setine bağlı bir artefakt mı?  ║
# ║  Kritik nokta: referans model HER KOLDA nedensel (y için sadece <y).     ║
# ║  Böylece kollar arasında değişen TEK şey özellik seti oluyor.            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import sys, subprocess
# river yalnızca ADWIN için gerekli. --no-deps kritik: numpy/scipy/pandas'a
# dokunmaz, dolayısıyla Colab'daki ikili uyumu bozmaz (ImportError: _center).
try:
    from river.drift import ADWIN
except ImportError:
    print("river kuruluyor (--no-deps, numpy'a dokunulmuyor) ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "-q", "--no-deps", "river"])
    from river.drift import ADWIN
print("river hazır.")

import os, re, json, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats as sstats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
import lightgbm as lgb

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    ROOT = "/content/drive/MyDrive/MSHA_AEJ_v7"
except Exception:
    ROOT = "./MSHA_AEJ_v7"

OLD_PKL = "/content/drive/MyDrive/MSHA_pipeline/ckpt/s1_data.pkl"

def _resolve_source():
    """Ham veri kaynağı. Öncelik: EAAI koşusunun ham checkpoint'i — böylece
    üç kol da BİREBİR aynı veriden çıkar (259.443 satır x 59 sütun)."""
    if os.path.exists(OLD_PKL):
        print(f"  kaynak: {OLD_PKL}  ({os.path.getsize(OLD_PKL)/1e6:.0f} MB, pickle)")
        return ("pickle", OLD_PKL)
    for p in [os.environ.get("MSHA_ACCIDENTS", "Accidents.txt"),          # [repo] local copy
              "/content/drive/MyDrive/Accidents.txt",
              "/content/drive/MyDrive/NIOSHA veri raporlar/Accidents.txt"]:
        if os.path.exists(p):
            print(f"  kaynak: {p}  ({os.path.getsize(p)/1e6:.0f} MB, txt)")
            return ("txt", p)
    raise FileNotFoundError(
        f"Ne {OLD_PKL} ne de Accidents.txt bulundu. "
        "Zip'i aç ya da OLD_PKL yolunu düzelt.")

CELL_VERSION = "v7-freq-terms"
print(f"[{CELL_VERSION}] ham veri kaynağı çözülüyor ...")
SRC_KIND, SRC_PATH = _resolve_source()
SEED, MIN_YEAR, MAX_YEAR, WARMUP = 20260820, 2000, 2023, 6
MAX_TRAIN, N_TFIDF, ADWIN_DELTA = 60_000, 100, 0.002
SEVERE = 3
SEV_NAMES = {0: "L0_no_lost_time", 1: "L1_restricted",
             2: "L2_lost_time", 3: "L3_permanent_or_fatal"}

# ── ALAN SINIFLARI ─────────────────────────────────────────────────────────
POST_OUTCOME = {
    "DEGREE_INJURY_CD": "Hedefin sayısal yeniden kodlaması",
    "SCHEDULE_CHARGE": "30 CFR Part 50: ciddiyetin fonksiyonu",
    "DAYS_LOST": "Hedef sınıfları bununla tanımlanıyor",
    "DAYS_RESTRICT": "DAYS RESTRICTED ACTIVITY sınıfını tanımlıyor",
    "TRANS_TERM": "Vaka kapandıktan sonra doldurulur",
    "RETURN_TO_WORK_DT": "Olay anında mevcut değil",
}
SEVERITY_TRIGGERED = {
    "IMMED_NOTIFY": "50.10: bildirim yalnızca ölüm/hayati tehlikede zorunlu",
    "IMMED_NOTIFY_CD": "Aynı gerekçe",
    "INVEST_BEGIN_DT": "Soruşturma ciddiyete tepki olarak açılır",
}
ADMIN = {k: "İdari / saf zaman" for k in
         ["DOCUMENT_NO", "CLOSED_DOC_NO", "MINE_ID", "CONTROLLER_ID",
          "CONTROLLER_NAME", "OPERATOR_ID", "OPERATOR_NAME", "CONTRACTOR_ID",
          "CAL_YR", "CAL_QTR", "FISCAL_YR", "FISCAL_QTR"]}
BORDERLINE = {k: "Triyajda bilinir, ciddiyetle eş-belirlenir" for k in
              ["NATURE_INJURY", "NATURE_INJURY_CD", "INJ_BODY_PART",
               "INJ_BODY_PART_CD", "NO_INJURIES"]}

# Kol tanımı: hangi grup DAHİL edilecek
ARMS = {
    "S": {"borderline": False, "post": False,
          "label": "Strict (point-of-incident only)"},
    "B": {"borderline": True,  "post": False,
          "label": "Borderline (+ triage-known clinical)"},
    "L": {"borderline": True,  "post": True,
          "label": "Leaky (EAAI feature set)"},
}

# ── NARRATİF OUTCOME SÖZLÜĞÜ — MSHA kısaltmaları dahil ────────────────────
OUTCOME_STEMS = [
    # ölüm
    "fatal", "died", "death", "deceas", "kill", "dead", "doa", "coroner",
    "autops", "expired", "pronounced",
    # hastane / nakil
    "hospital", "hosp", "admitted", "icu", "ambulanc", "airlift", "air lift",
    "life flight", "lifeflight", "medevac", "med evac", "transported",
    "transport to", "emergency room", "er for", "trauma center",
    # kalıcı hasar
    "amput", "amp of", "dismember", "paraly", "parapleg", "quadripleg",
    "permanent", "perm partial", "perm total", "disabilit", "disabl",
    "loss of use", "prosthe",
    # iş günü kaybı
    "days away", "days lost", "day lost", "lost time", "losttime", "lost day",
    "lost work", "time off", "off work", "off duty", "missed work",
    # kısıtlı görev
    "restricted duty", "restricted work", "restricted activity", "light duty",
    "lt duty", "lite duty", "modified duty", "restrict", "rtw",
    "return to work", "returned to work", "released to",
    # hafif
    "first aid", "firstaid", "first-aid", "no lost", "no treatment",
    "released", "no further", "refused treatment", "minor injury",
]
OUT_RE = re.compile("|".join(map(re.escape, OUTCOME_STEMS)))


def cpdir(arm):
    d = os.path.join(ROOT, f"arm_{arm}", "ckpt")
    os.makedirs(d, exist_ok=True)
    return d

def outdir(arm):
    d = os.path.join(ROOT, f"arm_{arm}")
    os.makedirs(d, exist_ok=True)
    return d

def save(arm, name, obj):
    p = os.path.join(cpdir(arm), f"{name}.pkl")
    with open(p + ".tmp", "wb") as f:
        pickle.dump(obj, f)
    os.replace(p + ".tmp", p)

def load(arm, name):
    p = os.path.join(cpdir(arm), f"{name}.pkl")
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f), True
        except Exception:
            os.remove(p)
    return None, False

# ══════════════════════════════════════════════════════════════════════════
# VERİ — kol başına özellik seti
# ══════════════════════════════════════════════════════════════════════════
SEVERITY_MAP = {
    "ALL OTHER CASES (INCL 1ST AID)": 0, "NO DYS AWY FRM WRK,NO RSTR ACT": 0,
    "DAYS RESTRICTED ACTIVITY ONLY": 1,
    "DAYS AWAY FROM WORK ONLY": 2, "DYS AWY FRM WRK & RESTRCTD ACT": 2,
    "PERM TOT OR PERM PRTL DISABLTY": 3, "FATALITY": 3,
}

_RAW_CACHE = {}

def build_arm(arm):
    res, hit = load(arm, "data")
    if hit:
        print(f"  ✅ arm {arm} data")
        return res

    if "df" not in _RAW_CACHE:
        print(f"  [{CELL_VERSION}] ham veri hazırlanıyor ...")
        if SRC_KIND == "pickle":
            with open(SRC_PATH, "rb") as f:
                obj = pickle.load(f)
            raw = (obj[0] if isinstance(obj, (tuple, list)) else obj).copy()
        else:
            raw = pd.read_csv(SRC_PATH, sep="|", encoding="latin-1",
                              on_bad_lines="skip", low_memory=False)
        raw.columns = [c.strip().upper() for c in raw.columns]
        # eski 12 sınıflı TARGET'ı at — hedefi DEGREE_INJURY'den yeniden kuruyoruz
        raw = raw.drop(columns=[c for c in ["TARGET"] if c in raw.columns])
        TXT = next(c for c in ["NARRATIVE", "ACCIDENT_DESCRIPTION"] if c in raw.columns)
        DTC = next(c for c in ["ACCIDENT_DT", "ACCIDENT_DATE"] if c in raw.columns)
        raw["_dt"] = pd.to_datetime(raw[DTC], errors="coerce")
        raw["YEAR"] = raw["_dt"].dt.year
        raw = raw.dropna(subset=["YEAR", "DEGREE_INJURY"])
        raw["YEAR"] = raw["YEAR"].astype(int)
        raw = raw[raw["YEAR"].between(MIN_YEAR, MAX_YEAR)]
        raw = raw.sort_values("_dt").reset_index(drop=True)
        sev = (raw["DEGREE_INJURY"].astype(str).str.strip().str.upper()
               .map({k.upper(): v for k, v in SEVERITY_MAP.items()}))
        print(f"  dışlanan (idari/karşılaştırılamaz): {int(sev.isna().sum()):,}")
        raw = raw[sev.notna()].copy()
        raw["TARGET"] = sev[sev.notna()].astype(int).values
        raw["_TXT"] = raw[TXT].fillna("").astype(str).str.lower()
        print(f"  kalan: {len(raw):,} satır  (S kolunda 217.305 çıkmıştı)")
        _RAW_CACHE.update(df=raw, TXT=TXT, DTC=DTC)

    raw, TXT, DTC = _RAW_CACHE["df"], _RAW_CACHE["TXT"], _RAW_CACHE["DTC"]
    cfg = ARMS[arm]

    excluded = dict(ADMIN)
    excluded.update(SEVERITY_TRIGGERED if not cfg["post"] else {})
    if not cfg["post"]:
        excluded.update(POST_OUTCOME)
    else:
        excluded["DEGREE_INJURY_CD"] = POST_OUTCOME["DEGREE_INJURY_CD"]  # hep dışarıda
    if not cfg["borderline"]:
        excluded.update(BORDERLINE)

    reserved = {"DEGREE_INJURY", "TARGET", "YEAR", "_dt", "_TXT", DTC, TXT}
    feats = [c for c in raw.columns if c not in excluded and c not in reserved]

    # narratif sözlük — BELGE FREKANSINA göre sıralanır (alfabetik DEĞİL).
    # v4'teki hata: get_feature_names_out() alfabetik döner, [:100] almak
    # "en sık 100" yerine "alfabetik ilk 100" veriyordu.
    if "terms" not in _RAW_CACHE:
        cv = CountVectorizer(ngram_range=(1, 2), min_df=20, max_features=2000,
                             stop_words="english", binary=True)
        M = cv.fit_transform(raw["_TXT"])
        nm = np.array(cv.get_feature_names_out())
        dfq = np.asarray(M.sum(axis=0)).ravel()
        _RAW_CACHE["terms"] = list(nm[np.argsort(-dfq)])   # sık -> seyrek
    ranked = _RAW_CACHE["terms"]
    audit = pd.DataFrame({"term": ranked, "df_rank": range(1, len(ranked) + 1)})
    audit["flagged"] = audit["term"].apply(lambda t: bool(OUT_RE.search(t)))
    if cfg["post"]:
        keep_terms = ranked[:N_TFIDF]
        audit["action"] = "retained (leaky arm)"
    else:
        keep_terms = [t for t in ranked if not OUT_RE.search(t)][:N_TFIDF]
        audit["action"] = np.where(audit["flagged"], "REMOVED", "retained")
    audit.to_csv(os.path.join(outdir(arm), "tfidf_vocab_audit.csv"), index=False)

    rows = []
    for c in raw.columns:
        if c in reserved:                  cls, why = "RESERVED", "hedef/zaman/metin"
        elif c in POST_OUTCOME:            cls, why = "POST_OUTCOME", POST_OUTCOME[c]
        elif c in SEVERITY_TRIGGERED:      cls, why = "SEVERITY_TRIGGERED", SEVERITY_TRIGGERED[c]
        elif c in ADMIN:                   cls, why = "ADMIN", ADMIN[c]
        elif c in BORDERLINE:              cls, why = "BORDERLINE", BORDERLINE[c]
        else:                              cls, why = "AT_INCIDENT", "İlk 7000-1 raporunda"
        rows.append({"field": c, "availability": cls,
                     "used_in_arm": c in feats, "justification": why})
    pd.DataFrame(rows).to_csv(
        os.path.join(outdir(arm), "feature_availability.csv"), index=False)

    d = raw[feats + ["_TXT", "TARGET", "YEAR"]].copy()
    out = (d, feats, keep_terms)
    save(arm, "data", out)
    print(f"  arm {arm}: {len(d):,} satır | {len(feats)} yapısal özellik | "
          f"{len(keep_terms)} tfidf | işaretli terim {int(audit.flagged.sum())}")
    return out

# ══════════════════════════════════════════════════════════════════════════
# BORU HATTI
# ══════════════════════════════════════════════════════════════════════════
class Prep:
    def __init__(self, feats, terms):
        self.F, self.T = feats, terms
        self.maps, self.num, self.cat = {}, [], []
        self.tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                  vocabulary=terms)
    def fit(self, d):
        sub = d[self.F]
        self.num = sub.select_dtypes(include=[np.number]).columns.tolist()
        self.cat = [c for c in self.F if c not in self.num]
        for c in self.cat:
            self.maps[c] = {v: i for i, v in
                            enumerate(sorted(sub[c].astype(str).fillna("NA").unique()))}
        self.tf.fit(d["_TXT"]); return self
    def transform(self, d):
        parts = []
        if self.num:
            parts.append(sp.csr_matrix(
                d[self.num].apply(pd.to_numeric, errors="coerce")
                           .fillna(-1).to_numpy(np.float32)))
        if self.cat:
            parts.append(sp.csr_matrix(np.column_stack([
                d[c].astype(str).fillna("NA").map(self.maps[c])
                    .fillna(-1).to_numpy(np.float32) for c in self.cat])))
        parts.append(self.tf.transform(d["_TXT"]).astype(np.float32))
        return sp.hstack(parts).tocsr()

LGB = dict(objective="multiclass", num_class=4, n_estimators=300,
           learning_rate=0.05, num_leaves=63, min_child_samples=30,
           subsample=0.8, colsample_bytree=0.8, is_unbalance=True,
           reg_alpha=0.1, reg_lambda=0.1, random_state=SEED,
           n_jobs=-1, verbose=-1)

def strat_sample(d, n):
    if len(d) <= n: return d
    return d.groupby("TARGET", group_keys=False).apply(
        lambda g: g.sample(max(1, int(round(n * len(g) / len(d)))),
                           random_state=SEED))

def fit_predict(train, test, feats, terms):
    train = strat_sample(train, MAX_TRAIN)
    t0 = time.perf_counter()
    p = Prep(feats, terms).fit(train)
    m = lgb.LGBMClassifier(**LGB).fit(p.transform(train), train["TARGET"].values)
    return m.predict(p.transform(test)), len(train), time.perf_counter() - t0

def row(strategy, y, start, n_tr, yt, yp, secs):
    m = yt == SEVERE
    return {"strategy": strategy, "year": int(y), "train_start": int(start),
            "n_train": int(n_tr), "n_test": int(len(yt)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(yt, yp)),
            "severe_recall": float((yp[m] == SEVERE).mean()) if m.any() else np.nan,
            "severe_as_minor": float((yp[m] < 2).mean()) if m.any() else np.nan,
            "fit_seconds": round(secs, 1)}

class DDM:
    def __init__(self, dl=3.0, min_n=30):
        self.dl, self.min_n = dl, min_n; self._r()
    def _r(self):
        self.n, self.p, self.s = 0, 1.0, 0.0
        self.pmin = self.smin = float("inf")
    def update(self, x):
        self.n += 1
        self.p += (x - self.p) / self.n
        self.s = (self.p * (1 - self.p) / self.n) ** 0.5
        if self.n < self.min_n: return False
        if self.p + self.s <= self.pmin + self.smin:
            self.pmin, self.smin = self.p, self.s
        d = (self.p + self.s) >= (self.pmin + self.dl * self.smin)
        if d: self._r()
        return bool(d)

def detect(A, order_seed=None, delta=ADWIN_DELTA):
    adwin, ddm = ADWIN(delta=delta), DDM()
    a, dd, log = [], [], []
    r = np.random.default_rng(order_seed) if order_seed is not None else None
    for y in sorted(A):
        e = A[y]["err"]
        if r is not None: e = e[r.permutation(len(e))]
        ai = di = None
        for i, v in enumerate(e):
            adwin.update(float(v))
            if adwin.drift_detected and ai is None: ai = i
            if ddm.update(int(v)) and di is None: di = i
        if ai is not None: a.append(int(y))
        if di is not None: dd.append(int(y))
        log.append({"year": int(y), "error_rate": float(e.mean()),
                    "adwin": ai is not None, "adwin_delay": ai,
                    "ddm": di is not None, "ddm_delay": di})
    return a, dd, sorted(set(a) & set(dd)), log


def run_arm(arm):
    print(f"\n{'='*70}\nKOL {arm} — {ARMS[arm]['label']}\n{'='*70}")
    d, feats, terms = build_arm(arm)
    years = sorted(d["YEAR"].unique())
    ev = [y for y in years if y >= MIN_YEAR + WARMUP]

    # ---- A: nedensel static-expanding + hata akışı -----------------------
    print(" [A] nedensel referans koşusu")
    for y in ev:
        if load(arm, f"A_{y}")[1]: continue
        tr, te = d[d.YEAR < y], d[d.YEAR == y]
        if len(tr) < 2000 or len(te) < 100: continue
        yp, n_tr, s = fit_predict(tr, te, feats, terms)
        yt = te["TARGET"].values
        save(arm, f"A_{y}", {"row": row("static_expanding", y, MIN_YEAR, n_tr, yt, yp, s),
                             "err": (yp != yt).astype(np.int8),
                             "y_true": yt.astype(np.int8), "y_pred": yp.astype(np.int8)})
        print(f"     {y}  F1={f1_score(yt, yp, average='macro', zero_division=0):.3f}"
              f"  err={float((yp != yt).mean()):.3f}  ({s:.0f}s)")
    A = {y: load(arm, f"A_{y}")[0] for y in ev if load(arm, f"A_{y}")[1]}

    # ---- B: kayma tespiti -------------------------------------------------
    print(" [B] kayma tespiti")
    a_y, d_y, cons, log = detect(A)
    pd.DataFrame(log).to_csv(os.path.join(outdir(arm), "drift_alarms.csv"), index=False)
    sens, hit = load(arm, "sens")
    if not hit:
        sens = {"ordering": [detect(A, order_seed=SEED + k)[2] for k in range(5)],
                "delta": [{"delta": dl, "consensus": detect(A, delta=dl)[2]}
                          for dl in [0.0005, 0.001, 0.002, 0.005, 0.01]]}
        save(arm, "sens", sens)
    print(f"     ADWIN={a_y}  DDM={d_y}  consensus={cons}")
    print(f"     sıralama={sens['ordering']}")
    print(f"     delta={[s['consensus'] for s in sens['delta']]}")

    # ---- C: stratejiler ---------------------------------------------------
    strats = ["rolling_5"] + (["adaptive_consensus"] if cons else [])
    if not cons:
        print("     (konsensüs yok → adaptive == static, atlanıyor)")
    print(" [C] stratejiler")
    for s_ in strats:
        for y in ev:
            tag = f"C_{s_}_{y}"
            if load(arm, tag)[1]: continue
            st = (y - 5) if s_ == "rolling_5" else \
                 (max([c for c in cons if c <= y - 1], default=MIN_YEAR))
            tr, te = d[(d.YEAR >= st) & (d.YEAR < y)], d[d.YEAR == y]
            if len(tr) < 2000 or len(te) < 100: continue
            yp, n_tr, sec = fit_predict(tr, te, feats, terms)
            yt = te["TARGET"].values
            save(arm, tag, {"row": row(s_, y, st, n_tr, yt, yp, sec),
                            "y_true": yt.astype(np.int8), "y_pred": yp.astype(np.int8)})
            print(f"     {s_:20s} {y}  F1={f1_score(yt, yp, average='macro', zero_division=0):.3f}")

    # ---- toplama ----------------------------------------------------------
    rows, preds = [], {}
    for y, v in A.items():
        rows.append(v["row"]); preds[("static_expanding", y)] = (v["y_true"], v["y_pred"])
    for s_ in strats:
        for y in ev:
            v, ok = load(arm, f"C_{s_}_{y}")
            if ok: rows.append(v["row"]); preds[(s_, y)] = (v["y_true"], v["y_pred"])
    py = pd.DataFrame(rows); py["arm"] = arm
    py.to_csv(os.path.join(outdir(arm), "per_year_results.csv"), index=False)

    for s_ in set(py["strategy"]):
        ks = [k for k in preds if k[0] == s_]
        yt = np.concatenate([preds[k][0] for k in ks])
        yp = np.concatenate([preds[k][1] for k in ks])
        pd.DataFrame(classification_report(
            yt, yp, labels=sorted(SEV_NAMES),
            target_names=[SEV_NAMES[i] for i in sorted(SEV_NAMES)],
            output_dict=True, zero_division=0)).T.to_csv(
            os.path.join(outdir(arm), f"per_class_{s_}.csv"))
        pd.DataFrame(confusion_matrix(yt, yp, labels=sorted(SEV_NAMES)),
                     index=[SEV_NAMES[i] for i in sorted(SEV_NAMES)],
                     columns=[SEV_NAMES[i] for i in sorted(SEV_NAMES)]).to_csv(
            os.path.join(outdir(arm), f"confusion_{s_}.csv"))

    with open(os.path.join(outdir(arm), "arm_manifest.json"), "w") as f:
        json.dump({"arm": arm, "label": ARMS[arm]["label"], "seed": SEED,
                   "n_rows": int(len(d)), "n_features": len(feats),
                   "tfidf_terms": terms, "adwin_years": a_y, "ddm_years": d_y,
                   "consensus": cons, "sensitivity": sens,
                   "msha_extraction_date": "BURAYA YAZ"}, f, indent=2, default=str)
    return py, cons


# ══════════════════════════════════════════════════════════════════════════
# ÇALIŞTIR
# ══════════════════════════════════════════════════════════════════════════
all_py, all_cons = [], {}
for arm in ["S", "B", "L"]:
    py, cons = run_arm(arm)
    all_py.append(py); all_cons[arm] = cons

# ── KOLLAR ARASI TABLO ────────────────────────────────────────────────────
big = pd.concat(all_py, ignore_index=True)
big.to_csv(os.path.join(ROOT, "all_arms_per_year.csv"), index=False)

cross = (big[big.strategy == "static_expanding"]
         .groupby("arm")
         .agg(mean_macro_f1=("macro_f1", "mean"),
              sd_macro_f1=("macro_f1", "std"),
              mean_balanced_acc=("balanced_acc", "mean"),
              mean_severe_recall=("severe_recall", "mean"),
              mean_error_rate=("macro_f1", lambda s: np.nan),
              n_years=("year", "count")).round(4))
cross["n_features"] = [
    json.load(open(os.path.join(outdir(a), "arm_manifest.json")))["n_features"]
    if os.path.exists(os.path.join(outdir(a), "arm_manifest.json"))
    else np.nan for a in cross.index]
cross["consensus_drift_years"] = [str(all_cons.get(a, "?")) for a in cross.index]
cross["label"] = [ARMS[a]["label"] for a in cross.index]
cross = cross.drop(columns=["mean_error_rate"])
cross.to_csv(os.path.join(ROOT, "TABLE_cross_arm.csv"))

print("\n" + "=" * 70)
print("  KOLLAR ARASI KARŞILAŞTIRMA  (nedensel static-expanding)")
print("=" * 70)
print(cross.to_string())
print(f"\n✅ bitti → {ROOT}/TABLE_cross_arm.csv")
