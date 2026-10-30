"""
Clinician-Reviewed Emergency Red Flag Rules
===========================================
Deterministic emergency detection rules designed to identify life-threatening
clinical warning signs without relying solely on generative model judgment.

Categories:
1. Cardiovascular / Myocardial Infarction (Heart Attack)
2. Cerebrovascular / Stroke (FAST Criteria)
3. Acute Respiratory Distress / Airway Compromise
4. Neurological Emergencies (Thunderclap Headache, Meningism)
5. Anaphylaxis (Severe Systemic Allergic Reaction)
6. Massive Hemorrhage & Acute Surgical Abdomen
7. Mental Health & Self-Harm Crisis
"""

import re
from typing import List, Dict, Any, Optional, Tuple


RED_FLAG_RULES: List[Dict[str, Any]] = [
    # ── 1. CARDIOVASCULAR / HEART ATTACK ─────────────────────────────────────
    {
        "rule_id": "CARD_01",
        "category": "Cardiovascular Emergency",
        "condition": "Suspected Acute Coronary Syndrome / Heart Attack",
        "patterns": [
            r"\b(heart\s*attack|cardiac\s*arrest|myocardial\s*infarction)\b",
            r"\b(crushing|squeezing|heavy|radiating|pressure)\s+(in\s+)?(my\s+)?(chest|breastbone)\b",
            r"\bchest\s+(pain|pressure|tightness|heaviness)\s+(spreading|radiating|shooting)\s+to\s+(my\s+)?(left\s+arm|jaw|neck|back|shoulder)\b",
            r"\b(severe|crushing)\s+chest\s+(pain|pressure)\b",
        ],
        "combined_required": [
            (
                r"\bchest\s+(pain|tightness|pressure|ache)\b",
                r"\b(short(ness)?\s+of\s+breath|sweat(ing|y)?|dizz(y|iness)|nausea|passed\s+out|faint(ing|ed)?)\b",
            )
        ],
        "urgency": "EMERGENCY",
        "action": "Immediate emergency dispatch (call 112 or 108 in India; 911/999 internationally).",
    },

    # ── 2. CEREBROVASCULAR / STROKE (FAST) ───────────────────────────────────
    {
        "rule_id": "STROKE_01",
        "category": "Neurological / Stroke Emergency",
        "condition": "Suspected Cerebrovascular Accident / Acute Stroke",
        "patterns": [
            r"\b(having\s+a\s+stroke|brain\s*stroke|mini\s*stroke|tia)\b",
            r"\b(face|mouth|eye)\s+(is\s+)?(droop(ing|ed)?|numb|sagging)\b",
            r"\b(can'?t|cannot|unable\s+to)\s+(speak|talk|form\s+words)\b",
            r"\b(slurr(ed|ing)\s+speech|speech\s+is\s+slurred)\b",
            r"\b(sudden(ly)?\s+)?(weakness|numbness|paralysis)\s+(in|on)\s+(one|my)\s+(side|arm|leg|face)\b",
            r"\b(can'?t|cannot)\s+(lift|raise|move)\s+(my\s+)?(left|right|one)\s+(arm|leg)\b",
        ],
        "urgency": "EMERGENCY",
        "action": "Time is critical (brain tissue loss). Call 112 or 108 immediately.",
    },

    # ── 3. ACUTE RESPIRATORY DISTRESS ─────────────────────────────────────────
    {
        "rule_id": "RESP_01",
        "category": "Respiratory Emergency",
        "condition": "Severe Acute Respiratory Distress / Airway Compromise",
        "patterns": [
            r"\b(can'?t|cannot|unable\s+to|struggling\s+to)\s+breathe?\b",
            r"\b(gasping\s+for\s+(air|breath)|severe\s+short(ness)?\s+of\s+breath)\b",
            r"\b(blue|purple|gray)\s+(lips|fingers|fingernails|face|skin)\b",
            r"\b(choking\s+on|airway\s+(is\s+)?blocked|stridor|suffocating)\b",
            r"\b(can'?t|cannot)\s+speak\s+in\s+full\s+sentences\b",
        ],
        "urgency": "EMERGENCY",
        "action": "Immediate acute oxygenation required. Call 112 or 108 immediately.",
    },

    # ── 4. THUNDERCLAP HEADACHE & MENINGISM ──────────────────────────────────
    {
        "rule_id": "NEURO_01",
        "category": "Acute Neurological Emergency",
        "condition": "Suspected Subarachnoid Hemorrhage or Acute Meningitis",
        "patterns": [
            r"\b(worst\s+headache\s+of\s+my\s+life|worst\s+pain\s+ever)\b",
            r"\bthunderclap\s+headache\b",
            r"\b(sudden(ly)?\s+)?(explosive|blinding|unbearable)\s+(onset\s+)?headache\b",
            r"\bheadache\s+reached\s+maximum\s+(in\s+)?(seconds|instantly)\b",
        ],
        "combined_required": [
            (
                r"\b(severe\s+headache|stiff\s+neck|neck\s+stiffness)\b",
                r"\b(high\s+fever|confusion|rash|sensitive\s+to\s+light|photophobia)\b",
            )
        ],
        "urgency": "EMERGENCY",
        "action": "Seek immediate emergency neurovascular evaluation. Call 112 or 108.",
    },

    # ── 5. ANAPHYLAXIS (SEVERE ALLERGIC REACTION) ───────────────────────────
    {
        "rule_id": "ALLERGY_01",
        "category": "Anaphylaxis Emergency",
        "condition": "Severe Systemic Allergic Reaction",
        "patterns": [
            r"\banaphylax(is|ic)\b",
            r"\b(throat|tongue|uvula|lips|mouth)\s+(is\s+)?swell(ing|ed)?\b",
            r"\b(throat\s+(is\s+)?clos(ing|ed)?|can'?t\s+swallow\s+or\s+breathe)\b",
        ],
        "combined_required": [
            (
                r"\b(stung\s+by|bee\s*sting|ate\s+peanuts|new\s+medication|allergic\s+reaction)\b",
                r"\b(hives|swelling|dizz(y|iness)|trouble\s+breathing|vomiting)\b",
            )
        ],
        "urgency": "EMERGENCY",
        "action": "Administer epinephrine autoinjector if prescribed and call 112 or 108 immediately.",
    },

    # ── 6. MASSIVE HEMORRHAGE & ACUTE ABDOMEN ────────────────────────────────
    {
        "rule_id": "HEM_01",
        "category": "Severe Hemorrhage / Acute Surgical Emergency",
        "condition": "Active Internal or Massive Bleeding",
        "patterns": [
            r"\b(cough(ing)?|vomit(ing)?)\s+(up\s+)?(large\s+amounts\s+of\s+)?blood\b",
            r"\b(vomit(ing)?\s+looks\s+like\s+coffee\s*grounds|coffee\s*ground\s+emesis)\b",
            r"\b(uncontrolled|heavy|spurting)\s+bleeding\b",
            r"\b(sudden|severe|unbearable)\s+sharp\s+abdominal\s+pain\s+(with\s+rigid\s+stomach|with\s+fainting)\b",
        ],
        "urgency": "EMERGENCY",
        "action": "Immediate emergency surgical and blood-loss care needed. Call 112 or 108.",
    },

    # ── 7. MENTAL HEALTH CRISIS / SELF-HARM ──────────────────────────────────
    {
        "rule_id": "CRISIS_01",
        "category": "Mental Health Crisis",
        "condition": "Acute Suicide / Self-Harm Risk",
        "patterns": [
            r"\b(want\s+to\s+(kill\s+myself|die|end\s+(it\s+all|my\s+life))|commit\s+suicide)\b",
            r"\b(thinking\s+(about|of)\s+suicide|suicidal\s+(thoughts|ideation|plan))\b",
            r"\b(harm(ing)?\s+myself|slit\s+my\s+wrists|take\s+all\s+my\s+pills)\b",
        ],
        "urgency": "EMERGENCY",
        "action": "Immediate crisis support. Contact Tele-MANAS at 14416 / 1800-891-4416 or call 112.",
    },
]


def match_red_flags(text: str) -> Optional[Dict[str, Any]]:
    """
    Evaluate user text against deterministic emergency red flag rules.
    Returns matched rule information or None if no deterministic red flag is found.
    """
    clean_text = text.lower().strip()

    for rule in RED_FLAG_RULES:
        # Check single primary patterns
        for pattern in rule.get("patterns", []):
            if re.search(pattern, clean_text, re.IGNORECASE):
                return {
                    "rule_id": rule["rule_id"],
                    "category": rule["category"],
                    "condition": rule["condition"],
                    "urgency": rule["urgency"],
                    "action": rule["action"],
                    "matched_pattern": pattern,
                }

        # Check combined patterns (pattern A AND pattern B must both match)
        for pat_a, pat_b in rule.get("combined_required", []):
            if re.search(pat_a, clean_text, re.IGNORECASE) and re.search(pat_b, clean_text, re.IGNORECASE):
                return {
                    "rule_id": rule["rule_id"],
                    "category": rule["category"],
                    "condition": rule["condition"],
                    "urgency": rule["urgency"],
                    "action": rule["action"],
                    "matched_pattern": f"{pat_a} + {pat_b}",
                }

    return None


# ── CONTEXTUAL INTENT CLASSIFICATION PATTERNS ──────────────────────────────
ACADEMIC_INTENT_PATTERNS: List[str] = [
    r"\b(what\s+is|what\s+are|how\s+to|how\s+do|how\s+is|explain|describe|discuss|outline|summarize|differentiate|compare|define)\b",
    r"\b(mechanism\s+of|pathophysiology\s+of|pathogenesis\s+of|etiology\s+of|causes\s+of|risk\s+factors\s+for)\b",
    r"\b(protocol\s+for|guidelines\s+for|algorithm\s+for|management\s+of|treatment\s+of|approach\s+to)\b",
    r"\b(investigation\s+of\s+choice|gold\s+standard|first[-\s]line|drug\s+of\s+choice|criteria\s+for|score|scoring|staging|classification)\b",
    r"\b(mbbs|cbme|neet[-\s]?pg|next|usmle|step\s*1|step\s*2|prof|profs|viva|short\s+note|long\s+question|high[-\s]?yield|pearls?|flashcard|mcq)\b",
    r"\b(vignette|clinical\s+case|case\s+presentation|case\s+study|scenario|hypothetical)\b",
    r"\b(a\s+\d{1,2}[-\s]year[-\s]old\s+(male|female|man|woman|boy|girl|patient|infant|neonate)|a\s+patient\s+(presents|presented|arrives|brought|admitted)|patient\s+with\s+history\s+of)\b",
]

FIRST_PERSON_DISTRESS_PATTERNS: List[str] = [
    r"\b(i('m| am)?\s+(having|experiencing|feeling|suffering|getting|in\s+severe)|i\s+(cannot|can'?t|collapsed|fainted))\b",
    r"\bmy\s+(chest|breastbone|left\s+arm|face|head|throat|stomach|breathing|vision|speech|mouth|uvula|tongue)\b",
    r"\b(help\s+me|hurts?\s+(so\s+)?bad|worst\s+pain\s+(i've|i\s+ever|of\s+my\s+life)|i\s+feel\s+like\s+i('m|\s+am)?\s+dying)\b",
    r"\b(can'?t|cannot|unable\s+to)\s+(breathe|catch\s+my\s+breath|speak|talk|swallow|move\s+my)\b",
    r"\b(right\s+now|since\s+\d+\s+min|just\s+now|happening\s+to\s+me|it\s+started\s+\d+\s+minutes?\s+ago)\b",
]

THIRD_PERSON_DISTRESS_PATTERNS: List[str] = [
    r"\b(my\s+(father|mother|dad|mom|husband|wife|son|daughter|child|baby|friend|brother|sister|grandma|grandpa|roommate|neighbor))\s+(is|just|collapsed|can'?t|has|started)\b",
    r"\b(someone\s+(here|next\s+to\s+me)?\s+(collapsed|is\s+unconscious|is\s+choking|can'?t\s+breathe|is\s+having|passed\s+out))\b",
    r"\b(he\s+(collapsed|passed\s+out|is\s+unconscious|can'?t\s+breathe|is\s+choking|is\s+turning\s+blue|is\s+clutching\s+his\s+chest))\b",
    r"\b(she\s+(collapsed|passed\s+out|is\s+unconscious|can'?t\s+breathe|is\s+choking|is\s+turning\s+blue|is\s+clutching\s+her\s+chest))\b",
]


def detect_academic_intent(text: str) -> bool:
    """Check if the text represents an academic, curricular, or clinical scenario inquiry."""
    clean_text = text.lower().strip()
    return any(re.search(pat, clean_text, re.IGNORECASE) for pat in ACADEMIC_INTENT_PATTERNS)


def detect_first_person_distress(text: str) -> bool:
    """Check if text indicates first-person present-tense emergency symptoms."""
    clean_text = text.lower().strip()
    return any(re.search(pat, clean_text, re.IGNORECASE) for pat in FIRST_PERSON_DISTRESS_PATTERNS)


def detect_third_person_distress(text: str) -> bool:
    """Check if text indicates acute emergency symptoms reported about another real person."""
    clean_text = text.lower().strip()
    return any(re.search(pat, clean_text, re.IGNORECASE) for pat in THIRD_PERSON_DISTRESS_PATTERNS)


def classify_safety_context(text: str, red_flag_match: Optional[Dict[str, Any]] = None) -> Tuple[str, bool]:
    """
    Classify the safety intent and determine whether to trigger an immediate emergency blockade.

    Returns (intent_type, is_emergency) where intent_type is one of:
      - 'crisis_self_harm': Acute suicide/self-harm risk (always emergency)
      - 'first_person_emergency': Present-tense self-reported emergency
      - 'third_person_emergency': Emergency reported about another real person
      - 'mixed_intent': Emergency symptoms mentioned alongside theoretical questions
      - 'academic_emergency': Educational/academic inquiry into an emergency medical topic
      - 'ambiguous_emergency': Red flag matched without clear academic context
      - 'none': No emergency detected
    """
    if not red_flag_match:
        return "none", False

    clean_text = text.lower().strip()

    # Rule 1: Mental health / crisis is non-negotiable emergency
    if red_flag_match.get("rule_id") == "CRISIS_01":
        return "crisis_self_harm", True

    has_academic = detect_academic_intent(clean_text)
    has_first_person = detect_first_person_distress(clean_text)
    has_third_person = detect_third_person_distress(clean_text)

    # Rule 2: Mixed intent - user claims active symptoms while asking academic questions
    if (has_first_person or has_third_person) and has_academic:
        return "mixed_intent", True

    # Rule 3: Third person real emergency
    if has_third_person:
        return "third_person_emergency", True

    # Rule 4: First person real emergency
    if has_first_person:
        return "first_person_emergency", True

    # Rule 5: Clear academic discussion of emergency condition
    if has_academic:
        return "academic_emergency", False

    # Rule 6: Question format check (e.g. "Can a heart attack cause shoulder pain?" or "What causes stroke?")
    if clean_text.endswith("?") or clean_text.startswith(("can ", "does ", "is ", "why ", "when ")):
        return "academic_emergency", False

    # Rule 7: Ambiguous phrasing (e.g. "crushing chest pain radiating to left arm")
    return "first_person_emergency", True

