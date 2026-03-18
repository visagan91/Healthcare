from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
MASTER_PATH = Path(
    "data/master_healthcare_dataset.csv"
)  # <-- update if your master CSV has a different name

IMG_DIR = Path("images/xray_images")  # folder with renamed images xray_000001.png ...
OUT_PATH = Path("data/xray_labels_from_master.csv")

SEED = 42
rng = np.random.default_rng(SEED)

# =========================
# LOAD MASTER DATASET
# =========================
assert MASTER_PATH.exists(), f"Master CSV not found: {MASTER_PATH.resolve()}"
df = pd.read_csv(MASTER_PATH)

print("Loaded master dataset:", df.shape)

# =========================
# LOAD IMAGES (SOURCE OF TRUTH)
# =========================
img_paths = sorted(
    list(IMG_DIR.glob("*.png")) +
    list(IMG_DIR.glob("*.jpg")) +
    list(IMG_DIR.glob("*.jpeg"))
)

assert len(img_paths) > 0, f"No images found in {IMG_DIR.resolve()}"
print("Images found:", len(img_paths))

# =========================
# ALIGN ROWS (1 IMAGE PER ROW)
# =========================
n = min(len(df), len(img_paths))
df = df.iloc[:n].copy()
img_paths = img_paths[:n]

df["image_id"] = [p.name for p in img_paths]
df["image_path"] = [str(p) for p in img_paths]

# =========================
# IMAGING LABEL (normal / abnormal)
# =========================
if "imaging_label" in df.columns:
    df["imaging_label"] = (
        df["imaging_label"]
        .astype(str)
        .str.lower()
        .replace({
            "0": "normal",
            "1": "abnormal",
            "no finding": "normal",
            "finding": "abnormal",
        })
    )
else:
    if "risk_category" in df.columns:
        df["imaging_label"] = df["risk_category"].map(
            {"low": "normal", "medium": "abnormal", "high": "abnormal"}
        ).fillna("abnormal")
    else:
        df["imaging_label"] = rng.choice(
            ["normal", "abnormal"], size=n, p=[0.55, 0.45]
        )

df["imaging_label"] = df["imaging_label"].str.lower()
df["y_cnn"] = (df["imaging_label"] == "abnormal").astype(int)

# =========================
# TRAIN / TEST SPLIT
# =========================
if "data_split" not in df.columns:
    mask = rng.random(n) < 0.8
    df["data_split"] = np.where(mask, "train", "test")

# =========================
# FINDING LABELS (DERIVED FROM USER EXAMPLES)
# =========================
EXAMPLE_FINDING_STRINGS = [
    "Emphysema|Infiltration|Pleural_Thickening|Pneumothorax",
    "Cardiomegaly|Emphysema",
    "No Finding",
    "Atelectasis",
    "Cardiomegaly|Edema|Effusion",
    "Consolidation|Mass",
    "No Finding",
    "No Finding",
    "Effusion",
    "No Finding",
    "Consolidation|Effusion|Infiltration|Nodule",
]

def extract_unique_findings(example_strings):
    findings = set()
    for s in example_strings:
        for part in s.split("|"):
            part = part.strip()
            if part != "No Finding":
                findings.add(part)
    return sorted(findings)

FINDINGS_POOL = extract_unique_findings(EXAMPLE_FINDING_STRINGS)

print("Derived FINDINGS_POOL:")
print(FINDINGS_POOL)

def synth_findings(is_abnormal: bool) -> str:
    if not is_abnormal:
        return "No Finding"

    # number of findings per abnormal image
    k = rng.choice([1, 2, 3, 4], p=[0.55, 0.30, 0.12, 0.03])
    chosen = rng.choice(FINDINGS_POOL, size=k, replace=False)
    return "|".join(chosen)

df["finding_labels"] = df["y_cnn"].apply(lambda y: synth_findings(y == 1))

# =========================
# BUILD IMAGE LABELS CSV
# =========================
cols = []
for c in ["patient_id", "encounter_id", "data_split"]:
    if c in df.columns:
        cols.append(c)

cols += [
    "image_id",
    "image_path",
    "imaging_label",
    "y_cnn",
    "finding_labels",
]

labels_df = df[cols].copy()

# =========================
# FINAL SANITY CHECK
# =========================
missing_imgs = (~labels_df["image_path"].apply(lambda p: Path(p).exists())).sum()
assert missing_imgs == 0, f"{missing_imgs} image paths do not exist!"

# =========================
# SAVE
# =========================
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
labels_df.to_csv(OUT_PATH, index=False)

print("\n✅ Saved image labels CSV:")
print(OUT_PATH.resolve())
print("Shape:", labels_df.shape)

print("\nSample rows:")
print(labels_df.head(10).to_string(index=False))

print("\nTop finding label distributions:")
print(labels_df["finding_labels"].value_counts().head(10))
