"""Generate synthetic EHR patient records for diabetes classification.

Generates ≥ 500 records covering all four diabetes classes:
  - Type 1  (label=0): age 10-40,  bmi 18-28, glucose 140-300, insulin 0-15
  - Type 2  (label=1): age 40-80,  bmi 25-45, glucose 100-250, insulin 10-50
  - Gestational (label=2): age 20-40, bmi 22-35, glucose 95-200, insulin 5-30
  - Other   (label=3): age 20-70,  bmi 20-35, glucose 80-180,  insulin 5-25

Usage:
    python -m diabetes_identifier.utils.generate_mock_data
    python diabetes_identifier/utils/generate_mock_data.py
"""
from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Seed for reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Clinical note templates per class
# ---------------------------------------------------------------------------
_NOTES_TYPE1 = [
    "Patient presents with polyuria and polydipsia since childhood. [NAME] was diagnosed with Type 1 diabetes at age [ID]. Currently on insulin pump therapy. HbA1c elevated. Glucose levels fluctuating.",
    "Young patient with autoimmune diabetes. Anti-GAD antibodies positive. Requires exogenous insulin. Reports frequent hypoglycemic episodes. [DOB] noted in chart.",
    "Juvenile-onset diabetes mellitus. Patient [NAME] uses continuous glucose monitor. Ketoacidosis episode last year. Insulin-dependent since diagnosis.",
    "Type 1 DM confirmed by C-peptide deficiency. Patient on basal-bolus insulin regimen. Carbohydrate counting education provided. [ID] verified.",
    "Autoimmune destruction of beta cells confirmed. Patient presents with DKA. Initiated on IV insulin drip. Blood glucose critically elevated on admission.",
]

_NOTES_TYPE2 = [
    "Middle-aged patient with longstanding Type 2 diabetes. [NAME] reports poor dietary compliance. On metformin and glipizide. BMI elevated. Sedentary lifestyle noted.",
    "Patient presents for diabetes management follow-up. HbA1c 9.2%. Obesity noted. [DOB] on file. Counseled on weight loss and exercise. Considering adding GLP-1 agonist.",
    "Type 2 DM with insulin resistance. Patient [NAME] on oral hypoglycemics. Fasting glucose consistently above target. Peripheral neuropathy developing.",
    "Established Type 2 diabetic with hypertension and dyslipidemia. [ID] confirmed. Metformin dose increased. Referred to dietitian for medical nutrition therapy.",
    "Overweight patient with poorly controlled Type 2 diabetes. Retinopathy screening due. Foot exam performed. Microalbuminuria detected on urinalysis.",
]

_NOTES_GESTATIONAL = [
    "Pregnant patient at 26 weeks gestation. Gestational diabetes diagnosed via OGTT. [NAME] counseled on dietary modifications. Fasting glucose borderline elevated.",
    "G2P1 patient with gestational diabetes mellitus. [DOB] recorded. Blood glucose monitoring initiated. Insulin therapy started due to diet failure. Fetal growth appropriate.",
    "Prenatal visit for gestational diabetes management. Patient [NAME] reports good compliance with diet. Glucose logs reviewed. No insulin required at this time.",
    "Gestational DM diagnosed at 24 weeks. [ID] on file. Patient educated on postpartum diabetes risk. Glucose targets set. Weekly monitoring planned.",
    "Third trimester patient with gestational diabetes. Ultrasound shows macrosomia. Insulin dose adjusted. [NAME] referred to maternal-fetal medicine for co-management.",
]

_NOTES_OTHER = [
    "Patient with MODY (Maturity-Onset Diabetes of the Young) confirmed by genetic testing. [NAME] managed with sulfonylurea. Family history significant.",
    "Secondary diabetes due to chronic pancreatitis. [DOB] noted. Exocrine and endocrine pancreatic insufficiency. On insulin and pancreatic enzyme replacement.",
    "Steroid-induced hyperglycemia in patient on long-term corticosteroid therapy. [ID] verified. Sliding scale insulin initiated. Glucose monitoring four times daily.",
    "Latent autoimmune diabetes in adults (LADA). Patient [NAME] initially misclassified as Type 2. Anti-GAD positive. Transitioning to insulin therapy.",
    "Drug-induced diabetes secondary to antipsychotic medication. Metabolic syndrome present. [NAME] referred to endocrinology for further evaluation and management.",
]

_NOTES_BY_LABEL = {
    0: _NOTES_TYPE1,
    1: _NOTES_TYPE2,
    2: _NOTES_GESTATIONAL,
    3: _NOTES_OTHER,
}

# ---------------------------------------------------------------------------
# Per-class feature distributions  (uniform ranges)
# ---------------------------------------------------------------------------
_DISTRIBUTIONS = {
    #        age       bmi       glucose   insulin
    0: dict(age=(10, 40),  bmi=(18, 28), glucose=(140, 300), insulin=(0, 15)),
    1: dict(age=(40, 80),  bmi=(25, 45), glucose=(100, 250), insulin=(10, 50)),
    2: dict(age=(20, 40),  bmi=(22, 35), glucose=(95, 200),  insulin=(5, 30)),
    3: dict(age=(20, 70),  bmi=(20, 35), glucose=(80, 180),  insulin=(5, 25)),
}

# ---------------------------------------------------------------------------
# Label names
# ---------------------------------------------------------------------------
_LABEL_NAMES = {0: "Type 1", 1: "Type 2", 2: "Gestational", 3: "Other"}


def _sample_class(label: int, n: int) -> pd.DataFrame:
    """Sample *n* records for a given class label."""
    dist = _DISTRIBUTIONS[label]
    notes_pool = _NOTES_BY_LABEL[label]

    rng = np.random.default_rng(RANDOM_SEED + label)

    ages = rng.integers(dist["age"][0], dist["age"][1] + 1, size=n)
    bmis = rng.uniform(dist["bmi"][0], dist["bmi"][1], size=n).round(1)
    glucoses = rng.uniform(dist["glucose"][0], dist["glucose"][1], size=n).round(1)
    insulins = rng.uniform(dist["insulin"][0], dist["insulin"][1], size=n).round(2)
    notes = [notes_pool[i % len(notes_pool)] for i in range(n)]

    return pd.DataFrame(
        {
            "age": ages,
            "bmi": bmis,
            "glucose": glucoses,
            "insulin": insulins,
            "label": label,
            "notes": notes,
        }
    )


def generate(n_per_class: int = 130, output_path: str | None = None) -> pd.DataFrame:
    """Generate synthetic EHR data and optionally save to *output_path*.

    Args:
        n_per_class: Number of records per class (4 classes → total ≥ 500 when ≥ 125).
        output_path: If provided, saves the CSV to this path.

    Returns:
        A :class:`pd.DataFrame` with columns: age, bmi, glucose, insulin, label, notes.
    """
    frames = [_sample_class(label, n_per_class) for label in range(4)]
    df = pd.concat(frames, ignore_index=True)

    # Shuffle rows
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")

    return df


def main() -> None:
    # Resolve output path relative to the project root (two levels up from this file)
    _here = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(_here))
    output_path = os.path.join(_project_root, "diabetes_identifier", "data", "raw", "ehr_synthetic.csv")

    df = generate(n_per_class=130, output_path=output_path)

    # Quick sanity summary
    print(f"\nTotal records : {len(df)}")
    print("Class distribution:")
    for label, name in _LABEL_NAMES.items():
        count = (df["label"] == label).sum()
        print(f"  {name:15s} (label={label}): {count}")


if __name__ == "__main__":
    main()
