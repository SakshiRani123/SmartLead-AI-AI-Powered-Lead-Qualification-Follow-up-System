"""
lead_scoring.py - Lead Scoring, Sentiment & Requirement Extraction
------------------------------------------------------------------
Provides:
1. ML inference for inquiry category classification (Hot, Warm, Cold).
2. Lead Priority Scoring (0 - 100) using a multi-factor transparent formula:
   - ML Classification confidence & category weight (40%)
   - Budget tier alignment (25%)
   - Stated decision timeline / urgency (20%)
   - Inquiry sentiment & intent depth (15%)
3. Sentiment Analysis:
   - Uses a transparent rule/lexicon-based heuristic analyzer.
   - Clearly documented as heuristic to avoid false claims of deep-learning sentiment.
4. Key Requirement and Entity Extraction:
   - Extracts specific technical terms, constraints, and project scopes.
"""

import re
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Keyword lexicons for sentiment and urgency heuristics
POSITIVE_WORDS = {
    "ready", "immediate", "urgent", "approved", "excited", "interested",
    "scalable", "kickoff", "priority", "deploy", "mandate", "perfect",
    "quickly", "definitely", "budgeted", "secured", "allocated", "asap"
}

NEGATIVE_WORDS = {
    "complaint", "issue", "problem", "delay", "failed", "broken",
    "unhappy", "frustrated", "poor", "slow", "buggy", "expensive", "risk"
}

URGENCY_WEIGHTS = {
    "Immediate (within 1-2 weeks)": 20,
    "Urgent (under 1 month)": 16,
    "1-3 months": 10,
    "Next quarter": 6,
    "No rush / Just exploring": 2,
    "Someday in the future": 0
}

BUDGET_WEIGHTS = {
    "$50,000+": 25,
    "$25,000 - $50,000": 22,
    "$15,000 - $30,000": 18,
    "$10,000 - $25,000": 16,
    "$10,000 - $20,000": 14,
    "$5,000 - $10,000": 10,
    "Under $10,000": 6,
    "Under $5,000": 4,
    "Flexible / Undecided": 8,
    "No Budget / Free": 0
}

CATEGORY_BASE_SCORES = {
    "Hot": 40,
    "Warm": 24,
    "Cold": 6
}

TECH_KEYWORDS = [
    "AI", "Machine Learning", "ML", "LLM", "Generative AI", "NLP",
    "API", "Dashboard", "SaaS", "Cloud", "AWS", "Azure", "GCP",
    "DevOps", "Database", "CRM", "Pipeline", "Analytics", "Automation",
    "Web App", "Mobile App", "React", "Python", "FastAPI", "Docker", "Kubernetes"
]


def analyze_sentiment_heuristic(text: str) -> Tuple[float, str]:
    """
    Analyzes sentiment using a lexicon heuristic scoring approach.
    DISCLOSURE: This is a transparent rule-based heuristic, NOT a deep neural NLP model.
    Returns:
        score: Normalized float between -1.0 (very negative) and +1.0 (very positive)
        label: 'Positive', 'Neutral', or 'Negative'
    """
    if not text or not isinstance(text, str):
        return 0.0, "Neutral"

    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    if not tokens:
        return 0.0, "Neutral"

    pos_count = sum(1 for w in tokens if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in tokens if w in NEGATIVE_WORDS)

    net_score = (pos_count - neg_count) / max(len(tokens) * 0.15, 1.0)
    # Clip between -1.0 and 1.0
    normalized_score = max(-1.0, min(1.0, round(net_score, 2)))

    if normalized_score > 0.15:
        label = "Positive"
    elif normalized_score < -0.15:
        label = "Negative"
    else:
        label = "Neutral"

    return normalized_score, label


def extract_requirements(message: str, service: str) -> str:
    """
    Extracts key technical requirements, tools, and project targets from inquiry text.
    """
    found_tech = []
    lower_msg = message.lower()

    for kw in TECH_KEYWORDS:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, lower_msg):
            found_tech.append(kw)

    # Check for budget mentions or timeline numbers
    details = []
    if found_tech:
        details.append(f"Target Tech/Capabilities: {', '.join(sorted(set(found_tech)))}")
    
    if service:
        details.append(f"Domain Focus: {service}")

    # Check for timeline flags
    if "asap" in lower_msg or "immediate" in lower_msg or "week" in lower_msg:
        details.append("Urgency Flag: Rapid deployment requested")
    elif "demo" in lower_msg or "exploring" in lower_msg:
        details.append("Exploratory Flag: Requesting architectural consultation / demo")

    return "; ".join(details) if details else f"Standard inquiry for {service}"


def calculate_lead_priority(
    predicted_category: str,
    confidence_score: float,
    budget_tier: str,
    urgency_tier: str,
    sentiment_score: float,
    message: str
) -> Tuple[int, Dict[str, Any]]:
    """
    Computes an explainable 0 to 100 Lead Priority Score.
    Breakdown:
      - ML Category Contribution: up to 40 pts
      - Budget Alignment: up to 25 pts
      - Stated Urgency: up to 20 pts
      - Sentiment & Intent Depth: up to 15 pts
    """
    # 1. ML Category Weight (Max 40)
    base_cat = CATEGORY_BASE_SCORES.get(predicted_category, 15)
    ml_points = round(base_cat * max(0.5, confidence_score))
    ml_points = min(40, max(0, ml_points))

    # 2. Budget Weight (Max 25)
    # Fallback to closest match if exact string differs slightly
    budget_pts = BUDGET_WEIGHTS.get(budget_tier, 10)

    # 3. Urgency Weight (Max 20)
    urgency_pts = URGENCY_WEIGHTS.get(urgency_tier, 5)

    # 4. Sentiment & Message Depth (Max 15)
    # Positive sentiment yields up to 7 pts, length/detail yields up to 8 pts
    sentiment_bonus = max(0, int((sentiment_score + 1.0) * 3.5))  # 0 to 7 pts
    word_count = len(message.split())
    depth_pts = min(8, int(word_count / 10))  # 1 pt per 10 words, max 8
    engagement_pts = min(15, sentiment_bonus + depth_pts)

    total_score = ml_points + budget_pts + urgency_pts + engagement_pts
    final_score = int(min(100, max(0, total_score)))

    # Determine recommended operational action
    if final_score >= 75:
        action = "High Priority: Route to Senior Technical Lead; schedule discovery call within 2 hours."
    elif final_score >= 45:
        action = "Medium Priority: Send portfolio case studies & calendar link; follow up within 24 hours."
    else:
        action = "Low Priority: Send automated educational resources & add to bi-weekly newsletter nurturing."

    factor_breakdown = {
        "ml_classification_points": ml_points,
        "budget_points": budget_pts,
        "urgency_points": urgency_pts,
        "sentiment_and_depth_points": engagement_pts,
        "total_score": final_score,
        "recommended_action": action
    }

    return final_score, factor_breakdown


def evaluate_lead(
    name: str,
    email: str,
    service: str,
    budget: str,
    urgency: str,
    message: str,
    model_pipeline=None
) -> Dict[str, Any]:
    """
    Full evaluation pipeline for a new lead.
    Integrates ML prediction (or heuristic fallback if model not loaded),
    sentiment heuristic, requirement extraction, and priority scoring.
    """
    # 1. Predict Category with ML Pipeline
    predicted_category = "Warm"
    confidence = 0.70

    if model_pipeline is not None:
        try:
            pred = model_pipeline.predict([message])[0]
            predicted_category = str(pred)
            if hasattr(model_pipeline, "predict_proba"):
                probs = model_pipeline.predict_proba([message])[0]
                classes = list(model_pipeline.classes_)
                if predicted_category in classes:
                    confidence = float(probs[classes.index(predicted_category)])
        except Exception as e:
            print(f"Inference fallback due to: {e}")
            # Safe heuristic fallback
            if any(w in message.lower() for w in ["ready", "immediate", "approved", "asap", "sign"]):
                predicted_category = "Hot"
                confidence = 0.85
            elif any(w in message.lower() for w in ["free", "trial", "sample", "just browsing", "school"]):
                predicted_category = "Cold"
                confidence = 0.80
            else:
                predicted_category = "Warm"
                confidence = 0.65
    else:
        # Heuristic fallback if model not yet trained
        if any(w in message.lower() for w in ["ready", "immediate", "approved", "asap", "sign"]):
            predicted_category = "Hot"
            confidence = 0.85
        elif any(w in message.lower() for w in ["free", "trial", "sample", "just browsing", "school"]):
            predicted_category = "Cold"
            confidence = 0.80

    # 2. Sentiment Analysis (Rule-Based Heuristic)
    sent_score, sent_label = analyze_sentiment_heuristic(message)

    # 3. Requirement Extraction
    key_reqs = extract_requirements(message, service)

    # 4. Priority Scoring
    priority_score, breakdown = calculate_lead_priority(
        predicted_category, confidence, budget, urgency, sent_score, message
    )

    return {
        "name": name,
        "email": email,
        "service": service,
        "budget": budget,
        "urgency": urgency,
        "message": message,
        "predicted_category": predicted_category,
        "confidence_score": round(confidence, 2),
        "priority_score": priority_score,
        "sentiment_score": sent_score,
        "sentiment_label": sent_label,
        "key_requirements": key_reqs,
        "recommended_action": breakdown["recommended_action"],
        "breakdown": breakdown
    }


if __name__ == "__main__":
    # Test lead evaluation
    test_inquiry = "We need an immediate quote and SOW for AI Solutions. Budget is approved for $30,000 and we want kickoff next week."
    res = evaluate_lead("Karan Malhotra", "karan@example.com", "AI & Machine Learning Solutions", "$25,000 - $50,000", "Immediate (within 1-2 weeks)", test_inquiry)
    print("Lead Scoring Test Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
