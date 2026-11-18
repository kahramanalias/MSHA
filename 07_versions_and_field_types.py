# =============================================================================
# What Can Be Known When a Mining Injury Is Reported? Temporal Availability and
# Severity Prediction in 24 Years of MSHA Records (Applied Sciences, 2026)
# Author: Abdulgani Kahraman, Balikesir University
#
# Step 7. Prints the software versions, the numeric/categorical treatment of fields and the river
#   detector defaults reported in Table S3. Trains no model.
#
# Run order: 01 -> 02 -> (03) -> 04 -> 05 -> 06 -> 07 -> 08. See README.md.
# Written for Google Colab (each script was run as one cell). Outside Colab the
# scripts use the current folder as BASE; put Accidents.txt there.
# Comments and console messages are partly in Turkish; the code is unchanged
# from the runs reported in the paper except for the data-path lines marked
# "[repo]".
# =============================================================================

# MSHA — sürüm ve ön işleme bilgisi (Table S3 için)          [v23-versions]
# ~1 dk. Hiçbir modeli yeniden eğitmez; yalnızca bilgi basar.
import sys, os, pickle, inspect, subprocess, platform
try:
    import river  # noqa
except ImportError:
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "river"])
import numpy, pandas, sklearn, lightgbm, scipy
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    BASE = "/content/drive/MyDrive"
except Exception:
    BASE = "."
V7 = os.path.join(BASE, "MSHA_AEJ_v7")

print("=== SÜRÜMLER ===")
print("python", platform.python_version())
for m in [numpy, pandas, scipy, sklearn, lightgbm]:
    print(m.__name__, getattr(m, "__version__", "?"))
try:
    import river; print("river", river.__version__)
except Exception as e:
    print("river yok:", e)

print("\n=== TARİH VE ÖRNEK ALANLARIN TÜRÜ (arm_L data.pkl) ===")
with open(os.path.join(V7, "arm_L", "ckpt", "data.pkl"), "rb") as f:
    dL, featsL, _ = pickle.load(f)
num = set(dL[featsL].select_dtypes(include=[numpy.number]).columns)
for c in ["RETURN_TO_WORK_DT", "INVEST_BEGIN_DT", "DAYS_LOST", "SCHEDULE_CHARGE", "IMMED_NOTIFY",
          "ACCIDENT_TIME", "TOT_EXPER", "NATURE_INJURY", "OCCUPATION_CD"]:
    if c in dL.columns:
        s = dL[c]
        print(f"{c:20s} dtype={str(s.dtype):8s} model={'sayısal' if c in num else 'kategorik(tamsayı kod)'} "
              f"eksik={s.isna().mean():.3f} örnek={s.dropna().astype(str).head(3).tolist()}")
print("sayısal alan sayısı:", len(num), "| kategorik:", len(featsL) - len(num))
print("sayısal alanlar:", sorted(num))

print("\n=== RIVER DEDEKTÖR VARSAYILANLARI ===")
try:
    import river.drift as rd
    try:
        import river.drift.binary as rb
    except Exception:
        rb = None
    for name in ["ADWIN", "DDM", "EDDM", "HDDM_A", "HDDM_W", "PageHinkley", "KSWIN"]:
        cls = getattr(rd, name, None) or (getattr(rb, name, None) if rb else None)
        if cls is None:
            print(name, "bulunamadı"); continue
        sig = inspect.signature(cls.__init__)
        print(f"{name:12s} ({cls.__module__}) " + ", ".join(
            f"{k}={v.default!r}" for k, v in sig.parameters.items() if k != "self" and v.default is not inspect._empty))
except Exception as e:
    print("river okunamadı:", e)

print("\n=== LightGBM subsample_freq varsayılanı ===")
try:
    print(lightgbm.LGBMClassifier().get_params().get("subsample_freq"))
except Exception as e:
    print("okunamadı:", e)
print("\n✅ Bitti. Çıktının tamamını gönder.")
