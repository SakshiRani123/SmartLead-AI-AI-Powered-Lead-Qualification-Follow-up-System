# SmartLead AI – AI-Powered Lead Qualification & Follow-up System

<p align="center">
  <b>An Intelligent AI/ML-Based Lead Management and Customer Inquiry Analysis System</b>
</p>
Local Host: http://localhost:8501

## 1. Project Overview

SmartLead AI is an AI/ML-powered lead qualification and follow-up system designed to help businesses analyse customer inquiries, identify potential customers, and prioritize sales leads automatically.

The system uses Natural Language Processing (NLP) and Machine Learning techniques to analyse customer messages and classify leads into three categories: **Hot, Warm, and Cold**.

It also generates lead priority scores, analyses customer requirements, and creates personalized follow-up email suggestions. An interactive Streamlit dashboard allows business users to monitor leads and understand customer inquiry patterns.

This project is developed as part of the **AI/ML Engineer Intern Technical Assessment for Cynbit**.

## 2. Problem Statement

Businesses receive multiple customer inquiries through websites, contact forms, and other communication channels. Manually analysing every inquiry requires time and effort, making it difficult for sales teams to identify high-priority customers.

SmartLead AI addresses this challenge by providing an automated lead qualification system that analyses customer information and helps businesses prioritize their potential customers.

## 3. Objectives

- Automate customer inquiry analysis using NLP.
- Classify customer leads into Hot, Warm, and Cold categories.
- Generate lead priority scores between 0 and 100.
- Identify customer requirements and urgency.
- Generate personalized follow-up email suggestions.
- Provide an interactive dashboard for lead monitoring.
- Improve lead management and sales team productivity.

## 4. Key Features

### Lead Classification
Automatically categorizes customer inquiries into Hot, Warm, and Cold leads using a Machine Learning model.

### Lead Scoring
Assigns a priority score between 0 and 100 based on customer inquiry characteristics and lead information.

### NLP-Based Text Analysis
Uses TF-IDF to convert customer inquiry messages into numerical features for Machine Learning.

### Customer Inquiry Analysis
Analyses customer requirements, selected services, budget, and urgency.

### AI Follow-up Email Generator
Generates personalized follow-up email suggestions based on customer requirements and lead category.

### Interactive Dashboard
Displays total leads, lead categories, priority scores, charts, and customer information.

### Data Management
Supports customer lead storage, viewing, and CSV-based data handling.

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| TF-IDF | Text feature extraction |
| Logistic Regression | Lead classification |
| Streamlit | Web application |
| Plotly | Data visualization |
| SQLite | Database management |
| Joblib | Model serialization |

## 6. System Architecture

The SmartLead AI system follows this workflow:

1. Customer submits an inquiry.
2. Input information is collected and processed.
3. NLP techniques extract numerical features from the inquiry.
4. The trained ML model predicts the lead category.
5. A lead priority score is generated.
6. A personalized follow-up email suggestion is created.
7. Results are displayed on the Streamlit dashboard.

## 7. Project Structure

```text
SmartLead-AI/
│
├── app.py
├── train_model.py
├── lead_scoring.py
├── email_generator.py
├── database.py
├── requirements.txt
├── README.md
│
├── data/
│   └── leads.csv
│
├── models/
│   └── lead_model.pkl
│
├── reports/
│   ├── AI_Opportunity_Report.pdf
│   └── Technical_Approach_Report.pdf
│
└── notebooks/
    └── experimentation.ipynb
```

## 8. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/SmartLead-AI.git
```

### Step 2: Navigate to the Project Directory

```bash
cd SmartLead-AI
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

For Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Train the Machine Learning Model

```bash
python train_model.py
```

### Step 7: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 9. Machine Learning Approach

The project uses a Natural Language Processing pipeline combined with Logistic Regression.

**TF-IDF Vectorization:** Converts customer inquiry text into numerical feature representations.

**Logistic Regression:** Classifies customer inquiries into Hot, Warm, and Cold categories.

**Lead Scoring:** Generates a priority score based on customer information and inquiry characteristics.

**Model Evaluation:** Uses accuracy, precision, recall, F1-score, and confusion matrix to evaluate classification performance.

The initial prototype uses a synthetic labelled dataset for experimentation and demonstration. Its predictions represent inquiry-based lead priority, not verified customer conversion probability.

## 10. Expected Business Benefits

- Reduces manual lead analysis.
- Helps sales teams identify high-priority inquiries.
- Improves customer response organization.
- Supports faster follow-up communication.
- Provides better visibility into customer inquiry patterns.
- Creates opportunities for future CRM automation.

## 11. Future Enhancements

- Integration with CRM platforms.
- Advanced LLM-based customer inquiry understanding.
- Real-time website chatbot integration.
- Automated email delivery with authorization.
- Improved lead scoring using real historical conversion data.
- Multilingual customer inquiry support.
- Advanced predictive analytics.

## 12. Limitations

- The initial model uses synthetic sample data.
- Classification quality depends on training data quality.
- Lead scores are prototype priority indicators, not actual sales conversion probabilities.
- Real-world deployment requires validation using genuine business data.
- Email generation requires additional integration for advanced LLM capabilities.

## 13. Project Purpose

This project demonstrates how Artificial Intelligence, Natural Language Processing, and Machine Learning can be applied to improve lead management, customer inquiry analysis, and business workflow automation.

Developed as part of the AI/ML Engineer Intern Assessment for Cynbit.

---

**Developed by:** Sakshi Rani

**Project:** SmartLead AI

**Domain:** Artificial Intelligence & Machine Learning

**Organization:** Cynbit

**GitHub:** https://github.com/SakshiRani123/SmartLead-AI-AI-Powered-Lead-Qualification-Follow-up-System
