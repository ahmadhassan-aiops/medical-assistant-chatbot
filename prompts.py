"""Health-profile model, prompt engineering and safety scaffolding.

The assistant is explicitly NOT a diagnostic tool. It helps a patient understand
whether the therapy their physician has ALREADY prescribed is consistent with
their overall medical profile and widely accepted international guidelines, and
answers personalised questions about medicines, nutrition and lifestyle.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Patient health profile
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HealthProfile:
    # Demographics
    age: str = ""
    sex: str = ""
    weight_kg: str = ""
    height_cm: str = ""
    pregnancy: str = ""            # e.g. "Not applicable", "Pregnant (week 20)", "Breastfeeding"

    # Vitals
    bp: str = ""                   # "128/82 mmHg"
    heart_rate: str = ""           # bpm
    resp_rate: str = ""            # /min
    temperature: str = ""          # °C
    spo2: str = ""                 # %
    blood_glucose: str = ""        # mg/dL or mmol/L

    # History
    comorbidities: list[str] = field(default_factory=list)
    comorbidities_other: str = ""
    allergies: str = ""            # drug + other allergies
    past_history: str = ""         # surgeries, major past illnesses
    family_history: str = ""

    # Current picture
    lab_results: str = ""          # free text: eGFR, HbA1c, lipids, LFTs, CBC, electrolytes…
    current_medications: str = ""  # everything the patient currently takes (incl. OTC/supplements)

    # The therapy under review (physician-provided)
    current_diagnosis: str = ""
    prescribed_therapy: str = ""   # what the physician prescribed for the current diagnosis

    # Lifestyle
    smoking: str = ""
    alcohol: str = ""
    diet: str = ""
    activity: str = ""

    def bmi(self) -> str:
        try:
            w = float(self.weight_kg)
            h = float(self.height_cm) / 100.0
            if w > 0 and h > 0:
                return f"{w / (h * h):.1f}"
        except (ValueError, ZeroDivisionError):
            pass
        return ""

    # ---- completeness, used to nudge the patient and guide the model ----
    def completeness(self) -> float:
        core = [
            self.age, self.sex, bool(self.comorbidities) or self.comorbidities_other,
            self.allergies, self.current_medications, self.lab_results,
            self.current_diagnosis, self.prescribed_therapy,
        ]
        filled = sum(1 for c in core if (c if isinstance(c, bool) else str(c).strip()))
        return filled / len(core)

    def missing_core(self) -> list[str]:
        checks = {
            "age": self.age,
            "sex": self.sex,
            "comorbidities / medical history": bool(self.comorbidities) or self.comorbidities_other,
            "allergies": self.allergies,
            "current medications": self.current_medications,
            "lab results": self.lab_results,
            "current diagnosis": self.current_diagnosis,
            "physician-prescribed therapy": self.prescribed_therapy,
        }
        return [
            name for name, v in checks.items()
            if not (v if isinstance(v, bool) else str(v).strip())
        ]


COMORBIDITY_OPTIONS = [
    "Hypertension", "Type 2 diabetes", "Type 1 diabetes", "Dyslipidemia",
    "Coronary artery disease", "Heart failure", "Atrial fibrillation",
    "Chronic kidney disease", "COPD", "Asthma", "Hypothyroidism",
    "Hyperthyroidism", "Liver disease", "Peptic ulcer disease", "Depression",
    "Anxiety", "Osteoarthritis", "Rheumatoid arthritis", "Gout", "Anemia",
    "Cancer (active/history)", "Stroke / TIA", "Epilepsy", "GERD",
]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────
def profile_to_text(p: HealthProfile) -> str:
    """Render a profile into a compact, LLM-friendly block. Omits empty fields."""
    lines: list[str] = []

    def add(label: str, value: str):
        if value and str(value).strip():
            lines.append(f"- {label}: {value}")

    lines.append("### Demographics")
    add("Age", p.age)
    add("Sex", p.sex)
    add("Weight (kg)", p.weight_kg)
    add("Height (cm)", p.height_cm)
    if p.bmi():
        add("BMI", p.bmi())
    add("Pregnancy / breastfeeding", p.pregnancy)

    lines.append("\n### Vitals")
    add("Blood pressure", p.bp)
    add("Heart rate (bpm)", p.heart_rate)
    add("Respiratory rate (/min)", p.resp_rate)
    add("Temperature (°C)", p.temperature)
    add("SpO2 (%)", p.spo2)
    add("Blood glucose", p.blood_glucose)

    lines.append("\n### Medical history")
    combined = ", ".join(p.comorbidities)
    if p.comorbidities_other:
        combined = ", ".join(x for x in [combined, p.comorbidities_other] if x)
    add("Comorbidities", combined)
    add("Allergies", p.allergies)
    add("Past history (surgeries / major illness)", p.past_history)
    add("Family history", p.family_history)

    lines.append("\n### Current status")
    add("Lab results", p.lab_results)
    add("Current medications (all, incl. OTC/supplements)", p.current_medications)

    lines.append("\n### Therapy under review (provided by physician)")
    add("Current diagnosis", p.current_diagnosis)
    add("Prescribed therapy", p.prescribed_therapy)

    lines.append("\n### Lifestyle")
    add("Smoking", p.smoking)
    add("Alcohol", p.alcohol)
    add("Diet", p.diet)
    add("Physical activity", p.activity)

    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are **MedGuide**, a careful medical-information assistant that helps a patient \
understand their OWN, already-established care. You support health literacy and \
shared decision-making with the patient's real clinicians. You are a wrapper over a \
language model and are NOT a licensed medical professional.

## Your role (what you DO)
- Help the patient understand whether the therapy their physician has ALREADY \
prescribed appears consistent with their overall medical profile and with widely \
accepted international guidelines (e.g. WHO, NICE, ADA, ESC/AHA/ACC, KDIGO, GOLD, \
GINA, and comparable specialty bodies).
- Explain their medicines: purpose, how to take them, common/important side effects, \
interactions with their other drugs/supplements, and cautions relevant to their \
labs (e.g. renal/hepatic function), age, pregnancy status and comorbidities.
- Give personalised nutrition/food guidance and lifestyle advice grounded in their \
specific conditions, medications and labs.
- Flag possible mismatches, gaps, dose concerns, interactions or contraindications \
between the prescribed therapy and the profile — framed as things to RAISE WITH \
THEIR PHYSICIAN, with the guideline-based reasoning explained plainly.

## Your boundaries (what you DO NOT do)
- You do NOT make or confirm a diagnosis, and you do NOT tell the patient what \
condition they have.
- You do NOT tell the patient to start, stop, or change the dose of any medicine on \
their own. Any change must go through their prescriber. You may say "this is worth \
discussing with your doctor" and explain why.
- You do NOT invent labs, vitals, or history that were not provided. If a decision \
depends on missing information, say what is missing and ask for it.
- You do NOT replace emergency care or a real clinical evaluation.

## How to answer
1. Ground every answer in THIS patient's profile below — reference the specific \
comorbidities, labs, meds, allergies and vitals that matter to the question.
2. When you assess the prescribed therapy, state clearly whether it appears \
**consistent with**, **needs clarification against**, or **potentially conflicts \
with** accepted guidelines for a patient with this profile — and name the general \
guideline principle you're relying on. Be honest about uncertainty and about what \
you cannot verify without the physician's full reasoning.
3. Prefer plain language. Use short sections or bullets. Define jargon briefly.
4. Be specific and personalised — avoid generic disclaimers-as-content. One concise \
reminder to confirm with their clinician is enough per answer.
5. If information is missing that would change the answer, ASK for it rather than \
guessing.

## Safety / red flags
If the patient describes symptoms that could be an emergency (e.g. chest pain, \
severe shortness of breath, signs of stroke — FAST, anaphylaxis, severe bleeding, \
suicidal thoughts, sudden severe or worst-ever pain, altered consciousness, \
symptoms of severe hypo-/hyperglycaemia), stop and advise them to seek emergency \
care / call local emergency services immediately, before anything else.

Always be warm, non-judgemental and clear. You are a knowledgeable guide helping the \
patient have a better conversation with their own care team — not a substitute for it.
"""

INTRO_MESSAGE = (
    "👋 Hi, I'm **MedGuide**. I can help you understand your prescribed treatment, "
    "medicines, food and lifestyle **based on your own health profile** — and whether "
    "your current therapy lines up with accepted international guidelines.\n\n"
    "I don't diagnose or change your treatment — I help you have a more informed "
    "conversation with your doctor. **Fill in your profile on the left**, then ask me "
    "anything, for example:\n"
    "- *Is my blood-pressure medication a good fit given my kidney results?*\n"
    "- *Which foods should I avoid with my current medicines?*\n"
    "- *Are any of my medications likely to interact with each other?*\n"
    "- *What lifestyle changes would help most for my conditions?*"
)


def build_messages(
    profile: HealthProfile, history: list[dict], user_msg: str
) -> list[dict]:
    """Assemble the full message list sent to the model."""
    profile_block = profile_to_text(profile)
    missing = profile.missing_core()
    missing_note = (
        f"\n\nNOTE: The following core details are still missing and may limit how "
        f"specific you can be — ask for them if they matter to the question: "
        f"{', '.join(missing)}."
        if missing else ""
    )

    system = (
        SYSTEM_PROMPT
        + "\n\n---\n## PATIENT PROFILE\n"
        + profile_block
        + missing_note
    )

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})
    return messages
