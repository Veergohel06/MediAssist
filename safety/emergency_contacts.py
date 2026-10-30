"""
Emergency Contacts & Helplines Module
=====================================
Defines emergency helpline contacts with emphasis on India (National 112,
108 Ambulance, 102 Maternal/Child, Tele-MANAS) alongside international references.
"""

from typing import Dict, List, Any

# Primary Emergency Numbers for India
INDIA_EMERGENCY_CONTACTS: List[Dict[str, str]] = [
    {
        "service": "All-in-One National Emergency Helpline",
        "number": "112",
        "description": "Unified 24x7 emergency response for Police, Fire, and Ambulance across India.",
    },
    {
        "service": "Medical Emergency Ambulance",
        "number": "108",
        "description": "Toll-free emergency ambulance and critical patient transportation service.",
    },
    {
        "service": "Maternal & Child Health Ambulance",
        "number": "102",
        "description": "Free ambulance transport for pregnant mothers, newborns, and infants.",
    },
    {
        "service": "Tele-MANAS (Mental Health Helpline)",
        "number": "14416 / 1800-891-4416",
        "description": "Government of India 24x7 tele-counseling for emotional distress and mental wellness.",
    },
    {
        "service": "National Health Helpline (MoHFW)",
        "number": "1075",
        "description": "Toll-free national health information and public health queries.",
    },
]

# International General Reference Numbers
GLOBAL_EMERGENCY_REFERENCES: Dict[str, str] = {
    "India": "112 / 108",
    "United States / Canada": "911",
    "United Kingdom": "999 / 111",
    "European Union": "112",
    "Australia": "000",
}


def get_emergency_guidance_text(region: str = "India", intent: str = "first_person_emergency") -> str:
    """
    Format standard, prominent emergency guidance instructions adapted to the user's intent:
    - first_person_emergency: User self-reporting acute symptoms
    - third_person_emergency: User reporting acute distress about another real person
    - mixed_intent: Acute distress mentioned alongside academic questions
    - crisis_self_harm: Acute mental health crisis / suicidal ideation
    """
    if intent == "crisis_self_harm":
        return "\n".join([
            "🚨 **CRITICAL CRISIS SUPPORT NOTICE: YOU ARE NOT ALONE**",
            "If you are feeling overwhelmed, hopeless, or having thoughts of self-harm, please reach out for immediate support.",
            "",
            "**IMMEDIATE CRISIS HELPLINES:**",
            "1. **Tele-MANAS (India - 24/7 Toll-Free):** Dial **14416** or **1800-891-4416**",
            "2. **KIRAN Mental Health Helpline (India):** Dial **1800-599-0019**",
            "3. **National Emergency Services:** Dial **112** (India) or **988** (US/Canada Suicide & Crisis Lifeline)",
            "4. Please reach out right now to a trusted family member, mentor, or mental health professional.",
            "",
            "**MediAssist is an automated educational tool and cannot provide clinical crisis intervention. Please connect with the human support services above.**",
        ])

    if intent == "third_person_emergency":
        return "\n".join([
            "🚨 **CRITICAL MEDICAL EMERGENCY NOTICE: ASSISTING SOMEONE IN DISTRESS**",
            "The symptoms you described for this person indicate a potential life-threatening medical emergency.",
            "",
            "**IMMEDIATE ACTIONS REQUIRED:**",
            "1. **Do not delay.** Call emergency medical services immediately on their behalf.",
            "2. If you are in **India**, dial **112** (National Emergency Helpline) or **108** (Emergency Ambulance) right away.",
            "3. If in another country, call your local emergency service (e.g., 911, 999, or 112).",
            "4. **Stay with the person:** Keep them calm and still. Do not leave them unattended.",
            "5. **Airway & Positioning:** If breathing and unconscious, place in the recovery position (on their side) unless spinal trauma is suspected. Do NOT give them anything to eat or drink.",
            "",
            "**MediAssist is an educational tool and CANNOT dispatch emergency ambulances or contact emergency personnel.**",
        ])

    if intent == "mixed_intent":
        return "\n".join([
            "🚨 **URGENT MEDICAL WARNING: ACUTE DISTRESS DETECTED**",
            "Your message mentions acute emergency symptoms alongside an academic inquiry.",
            "",
            "**SAFETY FIRST:**",
            "If you or someone with you is **actively experiencing these symptoms right now**, stop reading and seek emergency medical care immediately.",
            "- In **India**, dial **112** or **108** immediately.",
            "- Internationally, dial **911**, **999**, or **112**.",
            "- Do not delay clinical evaluation or attempt to drive yourself to the hospital.",
            "",
            "While theoretical medical questions can be studied later, acute clinical symptoms require immediate in-person emergency evaluation.",
            "**MediAssist is an educational learning assistant and CANNOT provide emergency care or dispatch assistance.**",
        ])

    # Default: first_person_emergency / ambiguous_emergency
    return "\n".join([
        "🚨 **CRITICAL MEDICAL EMERGENCY NOTICE**",
        "The symptoms you described may indicate a potentially life-threatening medical emergency.",
        "",
        "**IMMEDIATE ACTIONS REQUIRED:**",
        "1. **Do not wait or delay.** Call emergency medical services immediately.",
        "2. If you are in **India**, dial **112** (National Unified Helpline) or **108** (Emergency Ambulance) right away.",
        "3. If you are in another country, call your local emergency service (e.g., 911, 999, or 112).",
        "4. If safe to do so, alert a family member, neighbor, or coworker to assist you until professional responders arrive.",
        "5. **Do not attempt to drive yourself** to an emergency room if you are experiencing severe chest pain, shortness of breath, dizziness, or confusion.",
        "",
        "**MediAssist is an educational tool and CANNOT provide emergency dispatch, diagnosis, or clinical intervention.**",
    ])

