"""MedGuide — a personalised medical-information assistant (Streamlit UI).

Run:  streamlit run app.py
"""

from __future__ import annotations

# Trust the OS certificate store so TLS-inspecting security software (e.g. Avast
# Web Shield) or corporate proxies don't break HTTPS to the API. Must run before
# any network client is created. Safe no-op if truststore isn't installed.
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

import streamlit as st
from dotenv import load_dotenv

from config import PROVIDERS, DEFAULT_PROVIDER
from llm import stream_chat
from prompts import (
    HealthProfile,
    COMORBIDITY_OPTIONS,
    INTRO_MESSAGE,
    build_messages,
)

load_dotenv()

st.set_page_config(
    page_title="MedGuide — Personalised Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1000px; }

      .mg-hero {
        background: linear-gradient(120deg, #0ea5a4 0%, #0369a1 100%);
        color: #fff; border-radius: 18px; padding: 1.4rem 1.6rem;
        margin-bottom: 1rem; box-shadow: 0 10px 30px -12px rgba(3,105,161,.45);
      }
      .mg-hero h1 { font-size: 1.55rem; margin: 0 0 .25rem 0; font-weight: 700; }
      .mg-hero p  { margin: 0; opacity: .92; font-size: .95rem; }

      .mg-disclaimer {
        background: #fff7ed; border: 1px solid #fed7aa; color: #9a3412;
        border-radius: 12px; padding: .7rem 1rem; font-size: .82rem;
        margin-bottom: 1rem; line-height: 1.45;
      }
      .mg-pill {
        display:inline-block; background:#ecfeff; color:#0e7490;
        border:1px solid #a5f3fc; border-radius:999px;
        padding:.1rem .6rem; font-size:.72rem; margin-right:.3rem;
      }
      section[data-testid="stSidebar"] { width: 380px !important; }
      section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
      .stChatMessage { border-radius: 14px; }
      .mg-caption { color:#64748b; font-size:.78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("provider_key", DEFAULT_PROVIDER)


init_state()


def read_profile() -> HealthProfile:
    g = st.session_state.get
    return HealthProfile(
        age=g("age", ""),
        sex=g("sex", ""),
        weight_kg=g("weight_kg", ""),
        height_cm=g("height_cm", ""),
        pregnancy=g("pregnancy", ""),
        bp=g("bp", ""),
        heart_rate=g("heart_rate", ""),
        resp_rate=g("resp_rate", ""),
        temperature=g("temperature", ""),
        spo2=g("spo2", ""),
        blood_glucose=g("blood_glucose", ""),
        comorbidities=g("comorbidities", []),
        comorbidities_other=g("comorbidities_other", ""),
        allergies=g("allergies", ""),
        past_history=g("past_history", ""),
        family_history=g("family_history", ""),
        lab_results=g("lab_results", ""),
        current_medications=g("current_medications", ""),
        current_diagnosis=g("current_diagnosis", ""),
        prescribed_therapy=g("prescribed_therapy", ""),
        smoking=g("smoking", ""),
        alcohol=g("alcohol", ""),
        diet=g("diet", ""),
        activity=g("activity", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — settings + health profile
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 MedGuide")

    with st.expander("🔐 AI provider & model", expanded=False):
        provider_key = st.selectbox(
            "Provider",
            options=list(PROVIDERS.keys()),
            format_func=lambda k: PROVIDERS[k].label,
            key="provider_key",
        )
        provider = PROVIDERS[provider_key]
        # provider-scoped key so a stale model from another provider never
        # ends up as an invalid selectbox value
        st.session_state["_active_model"] = st.selectbox(
            "Model", options=provider.models, key=f"model_{provider_key}"
        )
        st.slider("Response creativity", 0.0, 1.0, 0.3, 0.05, key="llm_temperature",
                  help="Lower = more conservative and factual.")

        if provider.api_key():
            st.success(f"API key detected ({provider.env_var}) ✓")
        else:
            st.error(f"No key found. Set **{provider.env_var}** in `.env`.")
            if provider.signup_url:
                st.markdown(f"[Get a free key →]({provider.signup_url})")

    st.markdown("#### 👤 Your health profile")
    st.caption("Everything stays in your session. The more you share, the more "
               "personalised the answers.")

    with st.expander("Demographics", expanded=True):
        c1, c2 = st.columns(2)
        c1.text_input("Age", key="age", placeholder="e.g. 58")
        c2.selectbox("Sex", ["", "Female", "Male", "Other"], key="sex")
        c1.text_input("Weight (kg)", key="weight_kg", placeholder="e.g. 78")
        c2.text_input("Height (cm)", key="height_cm", placeholder="e.g. 172")
        st.selectbox(
            "Pregnancy / breastfeeding",
            ["", "Not applicable", "Pregnant", "Breastfeeding", "Trying to conceive"],
            key="pregnancy",
        )

    with st.expander("Vitals (if known)"):
        c1, c2 = st.columns(2)
        c1.text_input("Blood pressure", key="bp", placeholder="128/82 mmHg")
        c2.text_input("Heart rate (bpm)", key="heart_rate", placeholder="e.g. 76")
        c1.text_input("SpO₂ (%)", key="spo2", placeholder="e.g. 98")
        c2.text_input("Temp (°C)", key="temperature", placeholder="e.g. 37.0")
        c1.text_input("Resp. rate (/min)", key="resp_rate", placeholder="e.g. 16")
        c2.text_input("Blood glucose", key="blood_glucose", placeholder="110 mg/dL")

    with st.expander("Medical history"):
        st.multiselect("Comorbidities", COMORBIDITY_OPTIONS, key="comorbidities")
        st.text_input("Other conditions", key="comorbidities_other",
                      placeholder="anything not listed above")
        st.text_area("Allergies (drugs & other)", key="allergies", height=68,
                     placeholder="e.g. Penicillin (rash), sulfa drugs")
        st.text_area("Past history (surgeries / major illness)", key="past_history",
                     height=68, placeholder="e.g. MI 2019, cholecystectomy 2015")
        st.text_area("Family history (optional)", key="family_history", height=68,
                     placeholder="e.g. father — early CAD")

    with st.expander("Labs & current medications"):
        st.text_area(
            "Lab results", key="lab_results", height=110,
            placeholder="eGFR 52, creatinine 1.4, HbA1c 8.1%, LDL 130, "
                        "K+ 4.6, ALT 30, Hb 13.2 …",
        )
        st.text_area(
            "Current medications (ALL, incl. OTC & supplements)",
            key="current_medications", height=110,
            placeholder="Metformin 1000mg BID, Lisinopril 10mg OD, "
                        "Atorvastatin 20mg nocte, Aspirin 81mg, Vit D …",
        )

    with st.expander("Diagnosis & prescribed therapy", expanded=True):
        st.text_area("Current diagnosis (from your physician)",
                     key="current_diagnosis", height=68,
                     placeholder="e.g. Type 2 diabetes with early CKD; hypertension")
        st.text_area("Therapy your physician prescribed (under review)",
                     key="prescribed_therapy", height=90,
                     placeholder="e.g. Start Empagliflozin 10mg OD; increase "
                                 "Lisinopril to 20mg OD")

    with st.expander("Lifestyle"):
        c1, c2 = st.columns(2)
        c1.selectbox("Smoking", ["", "Never", "Former", "Current"], key="smoking")
        c2.selectbox("Alcohol", ["", "None", "Occasional", "Regular"], key="alcohol")
        st.text_input("Diet", key="diet", placeholder="e.g. vegetarian, low-salt")
        st.text_input("Physical activity", key="activity",
                      placeholder="e.g. walks 30 min/day")

    profile = read_profile()
    pct = profile.completeness()
    st.progress(pct, text=f"Profile completeness: {int(pct*100)}%")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main — header, disclaimer, chat
# ─────────────────────────────────────────────────────────────────────────────
provider = PROVIDERS[st.session_state["provider_key"]]

st.markdown(
    """
    <div class="mg-hero">
      <h1>🩺 MedGuide — your personalised health companion</h1>
      <p>Understand your prescribed treatment, medicines, food and lifestyle —
      checked against accepted international guidelines, tailored to your profile.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mg-disclaimer">
      <b>⚠️ Important:</b> MedGuide provides general medical <i>information</i> to help you
      understand your existing care — it does <b>not</b> diagnose you and does <b>not</b>
      change your treatment. Never start, stop, or change a medication without your
      prescriber. In an emergency, call your local emergency number immediately.
    </div>
    """,
    unsafe_allow_html=True,
)

missing = profile.missing_core()
if missing:
    st.markdown(
        "<span class='mg-caption'>💡 For more personalised answers, consider adding: "
        + ", ".join(f"<span class='mg-pill'>{m}</span>" for m in missing)
        + "</span>",
        unsafe_allow_html=True,
    )

# Render conversation
if not st.session_state["messages"]:
    with st.chat_message("assistant", avatar="🩺"):
        st.markdown(INTRO_MESSAGE)

for msg in st.session_state["messages"]:
    avatar = "🩺" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Input
prompt = st.chat_input("Ask about your therapy, medicines, food or lifestyle…")

if prompt:
    if not provider.api_key():
        st.error(
            f"⚠️ No API key for {provider.label}. Add **{provider.env_var}** to your "
            "`.env` file (see the sidebar → *AI provider & model*), then reload."
        )
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    model = st.session_state.get("_active_model", provider.models[0])
    messages = build_messages(
        profile, st.session_state["messages"][:-1], prompt
    )

    with st.chat_message("assistant", avatar="🩺"):
        try:
            reply = st.write_stream(
                stream_chat(
                    provider,
                    model,
                    messages,
                    temperature=st.session_state.get("llm_temperature", 0.3),
                )
            )
        except Exception as e:  # surface API/config errors cleanly
            reply = None
            st.error(f"Something went wrong talking to {provider.label}:\n\n`{e}`")

    if reply:
        st.session_state["messages"].append({"role": "assistant", "content": reply})
