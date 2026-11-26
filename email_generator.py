"""
email_generator.py - Personalized Follow-up Email Generation
------------------------------------------------------------
Generates tailored, professional B2B follow-up emails based on lead score,
category (Hot/Warm/Cold), extracted requirements, and service domain.

Supports:
1. Robust Offline Rule-Based Template Generator (100% reliable, zero cost, no API keys).
2. Optional Google Gemini LLM API integration (when GEMINI_API_KEY is provided).
3. Explicit human-in-the-loop: Does NOT send emails automatically; drafts them for review.
"""

import os
from typing import Dict, Any, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def generate_offline_email(lead_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Offline deterministic email generator.
    Produces clean, professional, high-converting B2B emails tailored to the lead's category.
    """
    name = lead_data.get("name", "there").strip()
    first_name = name.split()[0] if name else "Valued Client"
    service = lead_data.get("service", "Digital Solutions")
    category = lead_data.get("predicted_category", "Warm")
    requirements = lead_data.get("key_requirements", f"custom {service}")
    urgency = lead_data.get("urgency", "flexible")
    budget = lead_data.get("budget", "Flexible")

    if category == "Hot":
        subject = f"Accelerating Your {service} Deployment – Cynbit Discovery Session"
        body = f"""Hi {first_name},

Thank you for contacting Cynbit regarding your upcoming {service} project.

We reviewed your requirements ({requirements}) and noticed your target deployment timeline ({urgency}). Our senior engineering team has delivered similar high-impact architectures with rapid time-to-market.

Because your timeline is immediate, we would love to schedule a focused 20-minute discovery call to:
1. Walk through your technical architecture and scope requirements.
2. Outline recommended milestones, deliverables, and resource allocation within your {budget} budget framework.
3. Share relevant enterprise case studies and an initial Statement of Work (SOW) outline.

Could you let us know what time works best for you this Thursday or Friday? Alternatively, you can pick a time directly on our calendar: https://calendly.com/cynbit/discovery

Looking forward to collaborating with you.

Warm regards,

Client Solutions Team
Cynbit Technologies
https://cynbit.com
"""

    elif category == "Warm":
        subject = f"Exploring {service} for Your Team – Cynbit Overview & Case Studies"
        body = f"""Hi {first_name},

Thank you for your interest in Cynbit's {service} capabilities.

We saw that you are evaluating solutions for {requirements} with a planned timeline of {urgency}. Choosing the right technology partner is a critical step, and we would be delighted to provide the information you need to make an informed decision.

Here is what we recommend as next steps:
• Review our curated case studies in {service}: https://cynbit.com/case-studies
• An introductory technical consultation to answer any architecture questions and explore budget ranges ({budget}).

Would you be open to a brief call next week to review your questions and goals?

Best regards,

Business Development Team
Cynbit Technologies
https://cynbit.com
"""

    else:  # Cold lead / exploratory
        subject = f"Resources & Architecture Guides for {service} – Cynbit"
        body = f"""Hi {first_name},

Thank you for reaching out to Cynbit!

We received your inquiry regarding {service}. While it sounds like you may still be in the early exploratory phase, we wanted to ensure you have full access to our technical resources and whitepapers.

Here are a few helpful resources to get started:
• Cynbit Tech Insights & Architecture Whitepapers: https://cynbit.com/insights
• Free Solution Evaluation Checklist: https://cynbit.com/checklist

Whenever you are ready to explore a formal project or have specific technical questions, our engineering team is here to assist.

Wishing you great success with your initiatives!

Sincerely,

Community & Partnerships Team
Cynbit Technologies
https://cynbit.com
"""

    return {
        "subject": subject,
        "body": body,
        "generator_type": "Offline Deterministic Template Engine (Rule-based)"
    }


def generate_llm_email(lead_data: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Generates a personalized follow-up email using Google Gemini LLM API.
    Falls back gracefully to offline template if API key is invalid or unavailable.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")

    if not key or not GEMINI_AVAILABLE:
        # Fall back to offline generator
        return generate_offline_email(lead_data)

    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
You are an expert enterprise sales and solutions architect at Cynbit Technologies.
Draft a professional, compelling, and courteous personalized follow-up email for a client inquiry.

Client Lead Information:
- Client Name: {lead_data.get('name')}
- Email: {lead_data.get('email')}
- Interested Service: {lead_data.get('service')}
- Stated Budget: {lead_data.get('budget')}
- Stated Urgency: {lead_data.get('urgency')}
- Lead Category: {lead_data.get('predicted_category')} (Score: {lead_data.get('priority_score')}/100)
- Sentiment: {lead_data.get('sentiment_label')}
- Extracted Requirements: {lead_data.get('key_requirements')}
- Client Inquiry Message: "{lead_data.get('message')}"

Guidelines:
1. Provide a professional Subject line and Email Body.
2. If Hot lead: Be proactive, propose a discovery call with calendar link, demonstrate urgency and readiness.
3. If Warm lead: Focus on providing relevant case studies, architecture advice, and soft call-to-action.
4. If Cold lead: Be polite, offer helpful resources, and keep doors open without pressure.
5. Do NOT invent internal company commitments or false claims.
6. Return format:
SUBJECT: <Subject line>
BODY:
<Email body>
"""
        response = model.generate_content(prompt)
        text = response.text

        if "SUBJECT:" in text and "BODY:" in text:
            parts = text.split("BODY:")
            subject = parts[0].replace("SUBJECT:", "").strip()
            body = parts[1].strip()
        else:
            lines = text.strip().split("\n")
            subject = lines[0].replace("Subject:", "").strip()
            body = "\n".join(lines[1:]).strip()

        return {
            "subject": subject,
            "body": body,
            "generator_type": "Gemini 1.5 Flash (LLM Enhanced)"
        }

    except Exception as e:
        print(f"LLM generation failed ({e}), falling back to offline template.")
        fallback = generate_offline_email(lead_data)
        fallback["generator_type"] = f"Offline Template (LLM Fallback: {str(e)[:60]}...)"
        return fallback


if __name__ == "__main__":
    sample = {
        "name": "Sarah Jenkins",
        "email": "sarah@acuity.com",
        "service": "AI & Machine Learning Solutions",
        "budget": "$25,000 - $50,000",
        "urgency": "Immediate (within 1-2 weeks)",
        "predicted_category": "Hot",
        "priority_score": 90,
        "key_requirements": "Target Tech: AI, Machine Learning; Domain Focus: AI & Machine Learning Solutions",
        "message": "We need an immediate quote for deploying an AI qualification system for our Q1 release."
    }
    result = generate_offline_email(sample)
    print("Subject:", result["subject"])
    print("\nBody:\n", result["body"])
