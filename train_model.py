"""
train_model.py - Model Training & Evaluation for SmartLead AI
------------------------------------------------------------
1. Generates a synthetic dataset of realistic B2B customer inquiries.
   * Clearly marked and documented as synthetic to prevent misleading claims.
2. Splits dataset into Train (80%) and Test (20%) sets BEFORE fitting TF-IDF
   to prevent data leakage.
3. Builds an end-to-end Scikit-learn Pipeline (TfidfVectorizer + LogisticRegression)
   with class weight balancing to address class imbalance.
4. Evaluates model performance: Accuracy, Precision, Recall, F1-Score,
   and saves a Confusion Matrix plot.
5. Serializes the trained pipeline and metadata using Joblib into `models/`.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / "leads.csv"
MODEL_PATH = MODELS_DIR / "lead_classifier_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"


def generate_synthetic_dataset(n_samples_approx: int = 300) -> pd.DataFrame:
    """
    Generates a realistic synthetic labeled dataset of customer inquiries.
    NOTE: This dataset is synthetic for educational, evaluation, and demonstration purposes.
    It does not claim real historical conversion rates.
    """
    np.random.seed(42)

    hot_templates = [
        "We are ready to start immediately. Budget is approved for ${budget} and we urgently need {service}.",
        "Looking for an immediate vendor for {service}. We have an executive mandate and need to deploy within 2 weeks. Budget: ${budget}.",
        "We have an allocated budget of ${budget} for {service}. Can we schedule an onboarding or kickoff call this week?",
        "Urgent requirement: our existing system failed and we need an enterprise {service} implementation ASAP.",
        "We have an approved budget of ${budget} and require {service} deployed by next month. Please send proposal and master service agreement.",
        "Ready to sign contract for {service}. Who can we speak with today regarding procurement and timeline?",
        "High priority project: we need full-stack {service} with turnkey integration. We have ${budget} secured for Q1.",
        "We need an immediate quote and SOW for {service}. Decision maker is ready to finalize this Friday."
    ]

    warm_templates = [
        "We are evaluating options for {service}. We are planning for next quarter with an estimated budget of ${budget}.",
        "Interested in learning more about your {service} offerings. Could you share case studies and typical price ranges?",
        "Exploring {service} for our team. We are currently gathering vendor information and will decide in 1 to 2 months.",
        "We might need {service} later this year. Can you send a product demo and general rate card?",
        "Our management is discussing adopting {service}. What are your onboarding timelines and technical prerequisites?",
        "We have an exploratory budget around ${budget} for {service}. Looking to see how your solutions compare to competitors.",
        "Could you provide a high-level overview and pricing tiers for {service}? We are planning for H2.",
        "We are interested in {service}. Need to check if it aligns with our current architecture before requesting a formal quote."
    ]

    cold_templates = [
        "Just browsing your website. What services do you offer?",
        "Can you send free samples or a free trial for your tools?",
        "Just looking around for school research on {service}. No budget currently.",
        "Checking if you offer free consulting or internships for {service}.",
        "Do you have free open-source templates or free community tier for {service}?",
        "Just curious about your company. Are you hiring any freelance contractors?",
        "Can you email me a general brochure? No immediate project or timeline.",
        "Just test submitting this contact form to see if it works."
    ]

    services = [
        "AI & Machine Learning Solutions",
        "Generative AI & LLM Integration",
        "Full-Stack Web & Mobile App Development",
        "Cloud Architecture & DevOps Automation",
        "Data Engineering & Analytics Dashboards",
        "Custom SaaS Platform Development"
    ]

    budgets_hot = ["$25,000 - $50,000", "$50,000+", "$15,000 - $30,000", "$40,000 - $80,000"]
    budgets_warm = ["$5,000 - $10,000", "$10,000 - $20,000", "Under $10,000", "Flexible / Undecided"]
    budgets_cold = ["No Budget / Free", "Under $1,000", "Undecided", "N/A"]

    urgency_hot = ["Immediate (within 1-2 weeks)", "Urgent (under 1 month)"]
    urgency_warm = ["1-3 months", "Next quarter"]
    urgency_cold = ["No rush / Just exploring", "Someday in the future"]

    first_names = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Rohan", "Neha", "Karan", "Pooja", "David", "Sarah", "Michael", "Elena", "Alex"]
    last_names = ["Sharma", "Verma", "Patel", "Gupta", "Mehta", "Nair", "Rao", "Singh", "Das", "Joshi", "Smith", "Johnson", "Brown", "Taylor", "Miller"]
    companies = ["FinTech Labs", "HealthSync", "RetailGenius", "Nexus Cloud", "Acuity Logistics", "OmniCommerce", "TechScale", "BrightData"]

    records = []

    # Hot Leads (~100)
    for _ in range(100):
        srv = np.random.choice(services)
        bgt = np.random.choice(budgets_hot)
        urg = np.random.choice(urgency_hot)
        tmpl = np.random.choice(hot_templates)
        msg = tmpl.format(service=srv, budget=bgt.replace("$", ""))
        fn = np.random.choice(first_names)
        ln = np.random.choice(last_names)
        comp = np.random.choice(companies)
        records.append({
            "name": f"{fn} {ln}",
            "email": f"{fn.lower()}.{ln.lower()}@{comp.lower().replace(' ', '')}.com",
            "service": srv,
            "budget": bgt,
            "urgency": urg,
            "message": msg,
            "category": "Hot"
        })

    # Warm Leads (~110)
    for _ in range(110):
        srv = np.random.choice(services)
        bgt = np.random.choice(budgets_warm)
        urg = np.random.choice(urgency_warm)
        tmpl = np.random.choice(warm_templates)
        msg = tmpl.format(service=srv, budget=bgt.replace("$", ""))
        fn = np.random.choice(first_names)
        ln = np.random.choice(last_names)
        comp = np.random.choice(companies)
        records.append({
            "name": f"{fn} {ln}",
            "email": f"{fn.lower()}.{ln.lower()}@{comp.lower().replace(' ', '')}.com",
            "service": srv,
            "budget": bgt,
            "urgency": urg,
            "message": msg,
            "category": "Warm"
        })

    # Cold Leads (~90)
    for _ in range(90):
        srv = np.random.choice(services)
        bgt = np.random.choice(budgets_cold)
        urg = np.random.choice(urgency_cold)
        tmpl = np.random.choice(cold_templates)
        msg = tmpl.format(service=srv)
        fn = np.random.choice(first_names)
        ln = np.random.choice(last_names)
        comp = np.random.choice(companies)
        records.append({
            "name": f"{fn} {ln}",
            "email": f"{fn.lower()}.{ln.lower()}@{comp.lower().replace(' ', '')}.com",
            "service": srv,
            "budget": bgt,
            "urgency": urg,
            "message": msg,
            "category": "Cold"
        })

    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def train_and_evaluate(dataset_csv_path: Path = DATASET_PATH) -> dict:
    """
    Loads/generates synthetic dataset, splits into Train and Test,
    trains Scikit-learn Pipeline (TF-IDF + Logistic Regression),
    computes evaluation metrics, and serializes the model.
    """
    print("\n--- [Step 1: Dataset Generation / Loading] ---")
    if not dataset_csv_path.exists():
        print(f"Dataset not found at {dataset_csv_path}. Generating synthetic dataset...")
        df = generate_synthetic_dataset()
        df.to_csv(dataset_csv_path, index=False)
        print(f"Synthetic dataset saved to {dataset_csv_path} (Total rows: {len(df)})")
    else:
        df = pd.read_csv(dataset_csv_path)
        print(f"Loaded existing dataset from {dataset_csv_path} (Total rows: {len(df)})")

    # Data validation
    required_cols = ["message", "category"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Dataset must contain '{col}' column.")

    X = df["message"].astype(str)
    y = df["category"].astype(str)

    print(f"\nClass Distribution:\n{y.value_counts().to_string()}")

    print("\n--- [Step 2: Train/Test Split (Preventing Data Leakage)] ---")
    # We split 80% Train, 20% Test before any TF-IDF vectorizer fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    print("\n--- [Step 3: Building Scikit-Learn Pipeline] ---")
    # Pipeline combines TF-IDF representation and Logistic Regression classifier
    # class_weight='balanced' ensures minority classes receive proportional penalty
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),        # Unigrams and bigrams (e.g. 'immediate vendor')
            max_features=2500,          # Focus on top 2500 informative tokens
            stop_words='english',       # Remove non-informative stop words
            lowercase=True
        )),
        ('classifier', LogisticRegression(
            C=1.5,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ])

    print("Fitting Pipeline on Training data...")
    pipeline.fit(X_train, y_train)
    print("Model training completed successfully.")

    print("\n--- [Step 4: Model Evaluation on Unseen Test Data] ---")
    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)
    
    labels = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Weighted F1-Score: {f1 * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(report_text)

    # Save metrics metadata
    metrics_summary = {
        "is_synthetic_dataset": True,
        "disclaimer": "This model is trained on synthetic labeled data for educational and internship demonstration. It does not reflect production customer conversion prediction.",
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "labels": labels,
        "confusion_matrix": cm,
        "detailed_report": report_dict
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved metrics summary to {METRICS_PATH}")

    print("\n--- [Step 5: Serializing Model Pipeline] ---")
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return metrics_summary


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Loads and returns the trained pipeline."""
    if not model_path.exists():
        print(f"Model file not found at {model_path}. Training a new model now...")
        train_and_evaluate()
    return joblib.load(model_path)


if __name__ == "__main__":
    train_and_evaluate()
