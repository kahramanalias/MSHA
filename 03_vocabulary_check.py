# =============================================================================
# What Can Be Known When a Mining Injury Is Reported? Temporal Availability and
# Severity Prediction in 24 Years of MSHA Records (Applied Sciences, 2026)
# Author: Abdulgani Kahraman, Balikesir University
#
# Step 3 (optional). Checks that the 100-term vocabulary selected only from records before each
#   evaluation year gives the same results; step 4 reuses these runs as a cache if they exist.
#   The comparison with older caches (MSHA_AEJ/ablation...) is skipped when those files are absent.
#
# Run order: 01 -> 02 -> (03) -> 04 -> 05 -> 06 -> 07 -> 08. See README.md.
# Written for Google Colab (each script was run as one cell). Outside Colab the
# scripts use the current folder as BASE; put Accidents.txt there.
# Comments and console messages are partly in Turkish; the code is unchanged
# from the runs reported in the paper except for the data-path lines marked
# "[repo]".
# =============================================================================

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MSHA — SÖZLÜK YALNIZCA GEÇMİŞTEN + ABLASYON TUTARLILIĞI  [v20-vocab]   ║
# ║      exec(open('/content/drive/MyDrive/msha_v20_vocab_cell.py').read())  ║
# ║                                                                          ║
# ║  1) Ablasyon önbelleği ile ana analiz (arm S) aynı kayıtlar mı?          ║
# ║     Ablasyon terim listesi ile ana analiz terim listesi aynı mı?         ║
# ║  2) 100 terimlik sözlük her yıl YALNIZCA y'den önceki kayıtlardan        ║
# ║     seçilirse sonuçlar değişiyor mu?                                     ║
# ║     Report-time: 2006-2023 (Tablo 6 ile karşılaştırma)                   ║
# ║     Triage:      2018-2023 (Tablo 4/5 ile karşılaştırma)                 ║
# ║  ~30-40 dk. Checkpoint'li: kopsa aynı hücreyi tekrar çalıştır.           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os, re, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score, average_precision_score
import lightgbm as lgb

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    BASE = "/content/drive/MyDrive"
except Exception:
    BASE = "."
V7 = os.path.join(BASE, "MSHA_AEJ_v7"); AJ = os.path.join(BASE, "MSHA_AEJ")
OUT = os.path.join(V7, "vocab_v20"); CK = os.path.join(OUT, "ckpt"); os.makedirs(CK, exist_ok=True)
SEED, SEVERE, MAX_TRAIN, N_TFIDF = 20260820, 3, 60_000, 100
print("[v20-vocab] →", OUT)

# ana analizle (arms hücresi) birebir aynı sonuç-terimi listesi
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

def load(p):
    with open(p, "rb") as f: return pickle.load(f)

def strat(d, n):
    if len(d) <= n: return d
    return d.groupby("TARGET", group_keys=False).apply(
        lambda g: g.sample(max(1, int(round(n * len(g) / len(d)))), random_state=SEED))

def vocab_from(texts):           # ana analizdeki yöntemin aynısı, yalnızca verilen metinlerle
    cv = CountVectorizer(ngram_range=(1, 2), min_df=20, max_features=2000,
                         stop_words="english", binary=True)
    M = cv.fit_transform(texts)
    nm = np.array(cv.get_feature_names_out()); dfq = np.asarray(M.sum(0)).ravel()
    ranked = list(nm[np.argsort(-dfq)])
    return [t for t in ranked if not OUT_RE.search(t)][:N_TFIDF]

def fit_predict(tr, te, feats, terms):
    num = tr[feats].select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in feats if c not in num]
    maps = {c: {v: i for i, v in enumerate(sorted(tr[c].astype(str).fillna("NA").unique()))} for c in cat}
    tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, vocabulary=terms).fit(tr["_TXT"])
    def T(d):
        parts = []
        if num: parts.append(sp.csr_matrix(d[num].apply(pd.to_numeric, errors="coerce").fillna(-1).to_numpy(np.float32)))
        if cat: parts.append(sp.csr_matrix(np.column_stack([d[c].astype(str).fillna("NA").map(maps[c]).fillna(-1).to_numpy(np.float32) for c in cat])))
        parts.append(tf.transform(d["_TXT"]).astype(np.float32)); return sp.hstack(parts).tocsr()
    m = lgb.LGBMClassifier(**LGB).fit(T(tr), tr["TARGET"].values)
    return m.predict_proba(T(te))

# ── 1) kayıt ve terim tutarlılığı ───────────────────────────────────────────
dS, featsS, termsS = load(os.path.join(V7, "arm_S", "ckpt", "data.pkl"))
print("\n=== 1) Ablasyon ile ana analiz aynı veriyi mi kullanıyor? ===")
for p in [os.path.join(AJ, "ablation", "ckpt", "base.pkl"), os.path.join(AJ, "ablation_v8", "ckpt", "base.pkl")]:
    if os.path.exists(p):
        dfA, STRICT, _, _ = load(p)
        a = dS.groupby(["YEAR", "TARGET"]).size(); b = dfA.groupby(["YEAR", "TARGET"]).size()
        same = a.reindex(b.index.union(a.index), fill_value=0).equals(b.reindex(b.index.union(a.index), fill_value=0))
        print(f"  ana analiz: {len(dS):,} kayıt | ablasyon: {len(dfA):,} kayıt | yıl×sınıf sayıları aynı: {same}")
        print(f"  yapısal alanlar: ana {len(featsS)} / ablasyon {len(STRICT)} | aynı küme: {set(featsS)==set(STRICT)} | aynı sıra: {list(featsS)==list(STRICT)}")
        break
else:
    print("  ⚠ ablasyon base.pkl bulunamadı")
tp = os.path.join(AJ, "tfidf_fix", "ckpt", "terms.pkl")
if os.path.exists(tp):
    TOP_ALL, TOP_CLEAN, flagged = load(tp)
    print(f"  terim listesi: ana {len(termsS)} / ablasyon {len(TOP_CLEAN)} | ortak {len(set(termsS)&set(TOP_CLEAN))} | aynı sıra: {list(termsS)==list(TOP_CLEAN)}")
    print(f"  ablasyon filtresiz liste ile filtreli liste aynı: {list(TOP_ALL)==list(TOP_CLEAN)}")

# ── 2) sözlük yalnızca geçmişten ────────────────────────────────────────────
print("\n=== 2) 100 terim her yıl yalnızca geçmiş kayıtlardan seçildiğinde ===")
plan = [("S", dS, featsS, termsS, list(range(2006, 2024)))]
dB, featsB, termsB = load(os.path.join(V7, "arm_B", "ckpt", "data.pkl"))
plan.append(("B", dB, featsB, termsB, list(range(2018, 2024))))
res = {}
for arm, d, feats, gterms, years in plan:
    recs = []
    for y in years:
        key = os.path.join(CK, f"{arm}_{y}.pkl")
        if os.path.exists(key):
            r = load(key)
        else:
            t0 = time.perf_counter()
            past = d[d.YEAR < y]
            terms_y = vocab_from(past["_TXT"])
            pr = fit_predict(strat(past, MAX_TRAIN), d[d.YEAR == y], feats, terms_y)
            r = {"year": y, "yt": d[d.YEAR == y]["TARGET"].values.astype(np.int8),
                 "yp": np.argmax(pr, 1).astype(np.int8), "p3": pr[:, 3].astype(np.float32),
                 "overlap": len(set(terms_y) & set(gterms))}
            with open(key + ".tmp", "wb") as f: pickle.dump(r, f)
            os.replace(key + ".tmp", key)
            print(f"  {arm} {y}  macroF1={f1_score(r['yt'], r['yp'], average='macro'):.4f}  "
                  f"ortak terim={r['overlap']}/100  ({time.perf_counter()-t0:.0f}s)")
        recs.append(r)
    res[arm] = recs

def pooled(recs):
    yt = np.concatenate([r["yt"] for r in recs]); yp = np.concatenate([r["yp"] for r in recs])
    p3 = np.concatenate([r["p3"] for r in recs])
    tp = int(((yp == SEVERE) & (yt == SEVERE)).sum()); fl = int((yp == SEVERE).sum()); ns = int((yt == SEVERE).sum())
    P = tp / fl if fl else 0; R = tp / ns if ns else 0
    return P, R, (2*P*R/(P+R) if P+R else 0), fl, ns - tp, average_precision_score(yt == SEVERE, p3)

print("\n" + "=" * 96 + "\n  KARŞILAŞTIRMA (global sözlük = makaledeki değerler; yerel = her yıl yalnızca geçmişten)\n" + "=" * 96)
# Report-time, 18 yıl
csv = os.path.join(V7, "all_arms_per_year.csv")
g = pd.read_csv(csv); g = g[(g.strategy == "static_expanding") & (g.arm == "S")].set_index("year").macro_f1
loc = pd.Series({r["year"]: f1_score(r["yt"], r["yp"], average="macro") for r in res["S"]})
dd = (loc - g.loc[loc.index])
print(f"  Report-time 2006-2023 macro F1: global {g.loc[loc.index].mean():.4f} | yerel {loc.mean():.4f} | "
      f"fark ort {dd.mean():+.4f} (min {dd.min():+.4f}, max {dd.max():+.4f})")
r18 = [r for r in res["S"] if r["year"] >= 2018]
P, R, F, fl, ms, ap = pooled(r18)
print(f"  Report-time 2018-2023 yerel: macroF1 {np.mean([f1_score(r['yt'], r['yp'], average='macro') for r in r18]):.4f} "
      f"L3 P={P:.3f} R={R:.3f} F1={F:.3f} işaret={fl} kaçan={ms} AP={ap:.3f}   (global: 0.461 / 0.427 / 0.069 / 0.119 / 89 / 510 / 0.201)")
P, R, F, fl, ms, ap = pooled(res["B"])
print(f"  Triage 2018-2023 yerel: macroF1 {np.mean([f1_score(r['yt'], r['yp'], average='macro') for r in res['B']]):.4f} "
      f"L3 P={P:.3f} R={R:.3f} F1={F:.3f} işaret={fl} kaçan={ms} AP={ap:.3f}   (global: 0.668 / 0.966 / 0.730 / 0.832 / 414 / 148 / 0.889)")
print(f"  ortak terim sayısı (S): min {min(r['overlap'] for r in res['S'])}, max {max(r['overlap'] for r in res['S'])}")
print(f"\n✅ Bitti. Çıktının TAMAMINI gönder.")
