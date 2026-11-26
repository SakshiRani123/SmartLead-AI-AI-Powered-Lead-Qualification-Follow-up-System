# Technical Approach Report: SmartLead AI

**Project Name:** SmartLead AI – AI-Powered Lead Qualification and Follow-up System  
**Client / Stakeholder:** Cynbit  
**Document Type:** Technical Architecture, ML Engineering & System Specification  
**Author:** Senior AI/ML Engineer & Technical Project Architect  
**Date:** November 2026  
**Status:** Validated & Production-Ready Prototype  

---

## 1. System Architecture & High-Level Design

SmartLead AI is engineered as a modular, decoupled Python system designed for maintainability, zero mandatory API dependencies, and high explainability.

```
+-----------------------------------------------------------------------------------+
|                            STREAMLIT UI (app.py)                                  |
|  - Executive Dashboard | Lead Qualification Form | CRM & Analytics | CSV Import   |
+-----------------------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------------------+
|                        SCORING & NLP ENGINE (lead_scoring.py)                     |
|  - Heuristic Sentiment Analyzer    - Requirement & Tech Keyword Extractor         |
|  - Multi-Factor Priority Scorer (0 - 100)                                         |
+-----------------------------------------------------------------------------------+
         |                                                 |
         v                                                 v
+-----------------------------+          +------------------------------------------+
|   ML PIPELINE (train_model) |          |      EMAIL ENGINE (email_generator.py)   |
|  - TfidfVectorizer (1,2)    |          |  - Offline Deterministic Templates       |
|  - Logistic Regression      |          |  - Optional Gemini LLM Fallback          |
+-----------------------------+          +------------------------------------------+
                 \                                     /
                  v                                   v
+-----------------------------------------------------------------------------------+
|                       SQLITE PERSISTENCE LAYER (database.py)                      |
|  - Leads Schema | Safe Multi-Filter Queries | Status Management | Bulk Ingestion  |
+-----------------------------------------------------------------------------------+
```

---

## 2. Dataset Engineering & Synthetic Data Disclosure

### 2.1 Transparency Statement
The training dataset (`data/leads.csv`, $N=300$) is synthetically generated to reflect real-world B2B inquiries across enterprise software, cloud, and AI engineering services. 
* It is explicitly disclosed as synthetic for educational, internship evaluation, and architectural demonstration purposes.
* It does not claim real customer conversion rates or commercial production metrics.

### 2.2 Class Balance & Stratification
- **Hot Leads:** High urgency, confirmed budget, executive mandate, immediate deployment timeline.
- **Warm Leads:** Exploratory interest, planning for next quarter, evaluating competitive vendors.
- **Cold Leads:** Student inquiries, free service requests, casual browsing with no budget.

---

## 3. Data Leakage Prevention & ML Pipeline Architecture

### 3.1 Leakage Prevention Protocol
A common pitfall in NLP pipelines is fitting vectorizers (e.g., TF-IDF or Word2Vec) across the entire dataset before splitting train and test sets. SmartLead AI strictly enforces:
1. `train_test_split` with 80% Train ($N=240$) and 20% Test ($N=60$), stratified across `category`.
2. Pipeline fitting exclusively on `X_train`. The test set vocabulary is treated as completely unseen out-of-sample data.

### 3.2 Scikit-Learn Pipeline Specification
```python
Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),        # Captures unigrams and key phrase pairs
        max_features=2500,         # Vocabulary cap to prevent overfitting
        stop_words='english',      # Standard English stopword removal
        lowercase=True
    )),
    ('classifier', LogisticRegression(
        C=1.5,
        max_iter=1000,
        class_weight='balanced',   # Penalizes misclassifications inversely to class frequency
        random_state=42
    ))
])
```

---

## 4. Explainable Lead Priority Scoring Formula (0 to 100)

Rather than treating lead priority as an uninterpretable "black-box" probability, SmartLead AI uses a transparent multi-dimensional weighted formulation:

$$\text{Priority Score} = P_{\text{ML}} + P_{\text{Budget}} + P_{\text{Urgency}} + P_{\text{Engagement}}$$

Where:
1. **$P_{\text{ML}}$ (Max 40 Points):** Base score determined by predicted category ($\text{Hot}=40$, $\text{Warm}=24$, $\text{Cold}=6$) multiplied by the classifier confidence $\text{Confidence} \in [0.5, 1.0]$.
2. **$P_{\text{Budget}}$ (Max 25 Points):** Calibrated to project size ($\$50\text{k}+ = 25$, $\$25\text{k}-\$50\text{k} = 22$, down to $\text{Free} = 0$).
3. **$P_{\text{Urgency}}$ (Max 20 Points):** Timeline score ($\text{Immediate} = 20$, $\text{Urgent} = 16$, $\text{1-3 Months} = 10$, $\text{Future} = 0$).
4. **$P_{\text{Engagement}}$ (Max 15 Points):** Derived from sentiment polarity bonus (up to 7 points) and inquiry detail depth (up to 8 points based on word length).

---

## 5. Sentiment & Requirement Extraction

### 5.1 Heuristic Sentiment Analysis
- Uses an explainable token-lexicon analyzer scoring polarity on a normalized scale from $-1.0$ (Negative) to $+1.0$ (Positive).
- **Explicit Disclosure:** It is clearly labeled as a rule-based heuristic to prevent misleading claims of deep neural sentiment extraction while preserving zero computational latency.

### 5.2 Key Requirement Extraction
- Identifies critical technical domains (AI, ML, LLM, Cloud, DevOps, React, Docker, Kubernetes) and flags deployment urgency or exploratory requests via regex boundary matching.

---

## 6. Email Generation & Prompt Engineering

### 6.1 Dual-Engine Architecture
1. **Offline Deterministic Template Engine (Default):**
   - High reliability, zero latency, zero API cost.
   - Distinct templates dynamically populated with client name, service, budget, urgency, and extracted requirements.
2. **LLM Engine (Optional Google Gemini 1.5 Flash):**
   - Enabled via user toggle when a Gemini API key is supplied.
   - Structured system prompt enforcing enterprise sales tone, customized value propositions, and calendar booking links.
   - Built-in automatic fallback to offline templates upon network or token failure.

### 6.2 Human-in-the-Loop Policy
Generated emails are populated into interactive text areas for human sales representative review, personalization, and final sign-off. The application strictly forbids unmoderated automated dispatch.

---

## 7. Model Evaluation & Benchmark Results

Evaluated on the held-out stratified test set ($N=60$):
- **Accuracy:** $100\%$ on synthetic evaluation benchmark
- **Precision:** $1.00$ (Hot), $1.00$ (Warm), $1.00$ (Cold)
- **Recall:** $1.00$ (Hot), $1.00$ (Warm), $1.00$ (Cold)
- **Weighted F1-Score:** $1.00$

*(Confusion matrix and metrics are persisted in `models/model_metrics.json` and visualized interactively in the Streamlit application).*

---

## 8. Deployment, Security & Production Roadmap

### 8.1 Deployment Architecture
- **Local / On-Premise:** Streamlit single-node deployment running on Python 3.10+ / 3.14.
- **Cloud Containerization:** Containerizable via Docker (Alpine / Debian Slim Python base).
- **Database:** SQLite (`data/leads.db`) for lightweight deployments; seamless drop-in transition to PostgreSQL via SQLAlchemy.

### 8.2 Security & Data Privacy
- **No Third-Party Data Transmission:** In offline mode, customer inquiries remain 100% on-premises.
- **Input Sanitization:** Parameterized SQL queries prevent SQL injection vulnerabilities.
- **API Key Safety:** Keys are handled in memory and never logged or serialized to disk.

### 8.3 Future Enhancements
1. Fine-tuning an embedding-based transformer (Sentence-Transformers / RoBERTa) on real historical CRM leads.
2. Direct bi-directional synchronization with HubSpot / Salesforce via Webhooks.
3. Automated multi-turn conversational follow-up sequencing.
