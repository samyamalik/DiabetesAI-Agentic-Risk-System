"""
Streamlit Frontend — Agentic AI Diabetes Risk System.
Patient-facing + clinician-style dashboard with full pipeline integration.
"""

import streamlit as st
import json
import sys
import io
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict
from fpdf import FPDF

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import (
    IngestionAgent, RiskAgent, ExplainabilityAgent,
    RecommendationAgent, MonitoringAgent,
)
from src.config.settings import MEDICAL_DISCLAIMER

# ── Constants ────────────────────────────────────────────────────────
RECORDS_PATH = PROJECT_ROOT / "data" / "patient_records.json"
RISK_COLORS = {
    "Low": "#22c55e", "Moderate": "#eab308",
    "High": "#f97316", "Very High": "#ef4444",
}

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesAI — Agentic Risk System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
#  CSS — Glassmorphism Design System
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{font-family:'Inter',sans-serif}

/* ── Global Background ── */
.stApp {
  background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
  min-height: 100vh;
}
.block-container{padding-top:1.5rem;max-width:1200px}

/* ── Global Text Color ── */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stApp .stMarkdown, .stApp .stText {color: #e2e8f0 !important}
h1, h2, h3, h4, h5, h6 {color: #ffffff !important}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: rgba(15, 12, 41, 0.85) !important;
  backdrop-filter: blur(14px);
  border-right: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {color: #cbd5e1 !important}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {color: #f1f5f9 !important}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {background: transparent; border-bottom:1px solid rgba(255,255,255,0.15)}
.stTabs [data-baseweb="tab"] {color: #94a3b8 !important; background:transparent}
.stTabs [aria-selected="true"] {color: #fff !important; border-bottom-color: #818cf8 !important}

/* ── Glass Card (parent) ── */
.glass-card {
  background: rgba(255,255,255,0.08);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.25);
  padding: 20px 24px;
  margin-bottom: 18px;
  transition: all 0.3s ease;
}
.glass-card:hover {
  transform: scale(1.015);
  border-color: rgba(255,255,255,0.25);
  box-shadow: 0 12px 40px rgba(0,0,0,0.35);
  background: rgba(255,255,255,0.11);
}
.glass-card h4 {color:#f1f5f9 !important; margin-top:0; margin-bottom:10px; font-weight:600}
.glass-card p, .glass-card li, .glass-card span {color:#cbd5e1 !important}

/* ── Glass Sub-Card (children inside Recommendations) ── */
.glass-sub {
  background: rgba(255,255,255,0.06);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 14px 18px;
  margin: 10px 0;
  transition: all 0.3s ease;
}
.glass-sub:hover {
  transform: scale(1.01);
  background: rgba(255,255,255,0.09);
}
.glass-sub h5 {color:#e2e8f0 !important; margin-top:0; margin-bottom:6px; font-size:0.95rem}

/* ── Risk badge ── */
.risk-badge{display:inline-block;padding:8px 22px;border-radius:24px;
  font-weight:700;font-size:1.05rem;color:#fff;letter-spacing:.3px}
.risk-low{background:#22c55e}.risk-moderate{background:#eab308;color:#111}
.risk-high{background:#f97316}.risk-veryhigh{background:#ef4444}

/* ── Alert boxes ── */
.alert-critical{background:rgba(239,68,68,0.15);border-left:4px solid #ef4444;
  padding:12px 16px;border-radius:0 10px 10px 0;margin:8px 0;color:#fca5a5 !important}
.alert-high{background:rgba(249,115,22,0.15);border-left:4px solid #f97316;
  padding:12px 16px;border-radius:0 10px 10px 0;margin:8px 0;color:#fdba74 !important}
.alert-medium{background:rgba(234,179,8,0.15);border-left:4px solid #eab308;
  padding:12px 16px;border-radius:0 10px 10px 0;margin:8px 0;color:#fde68a !important}

/* ── Agent status dot ── */
.agent-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.agent-dot.on{background:#22c55e}.agent-dot.warn{background:#eab308}

/* ── Disclaimer ── */
.disclaimer-box{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
  border-radius:12px;padding:14px 18px;font-size:.85rem;margin-top:1rem;color:#94a3b8 !important}

/* ── Download button ── */
.stDownloadButton > button {
  background: linear-gradient(135deg, #6366f1, #818cf8) !important;
  color: #fff !important; border: none !important; border-radius: 10px !important;
  font-weight: 600 !important; padding: 8px 18px !important;
  transition: all 0.3s ease !important; font-size: 0.9rem !important;
}
.stDownloadButton > button:hover {
  transform: scale(1.03) !important;
  box-shadow: 0 6px 24px rgba(99,102,241,0.45) !important;
  background: linear-gradient(135deg, #818cf8, #a5b4fc) !important;
}

/* ── Progress bar color override ── */
.stProgress > div > div > div > div {background: linear-gradient(90deg,#6366f1,#818cf8) !important}

/* ── Metric overrides ── */
[data-testid="stMetricValue"] {color:#ffffff !important; font-weight:700}
[data-testid="stMetricLabel"] {color:#94a3b8 !important}

/* ── Form & input styling ── */
.stNumberInput label, .stTextInput label {color:#cbd5e1 !important}
button[kind="primaryFormSubmit"], .stFormSubmitButton button {
  background: linear-gradient(135deg,#6366f1,#818cf8) !important;
  color:#fff !important; border:none !important; border-radius:10px !important;
  font-weight:600 !important; padding:10px 20px !important;
  transition: all 0.3s ease !important;
}
button[kind="primaryFormSubmit"]:hover, .stFormSubmitButton button:hover {
  transform: scale(1.02) !important;
  box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  DATA PERSISTENCE
# ══════════════════════════════════════════════════════════════════════
def load_all_records() -> Dict:
    if RECORDS_PATH.exists():
        return json.loads(RECORDS_PATH.read_text())
    return {}


def save_all_records(data: Dict):
    RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORDS_PATH.write_text(json.dumps(data, indent=2, default=str))


def load_patient(pid: str) -> list:
    return load_all_records().get(pid, [])


def save_record(pid: str, record: dict):
    all_data = load_all_records()
    all_data.setdefault(pid, []).append(record)
    save_all_records(all_data)


# ══════════════════════════════════════════════════════════════════════
#  AGENT INITIALISATION (cached)
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading AI agents …")
def init_agents():
    # Cache invalidated to load new encoders with 'Diabetic' label
    return {
        "risk": RiskAgent(),
        "explain": ExplainabilityAgent(),
        "recommend": RecommendationAgent(),
        "monitor": MonitoringAgent(),
    }


# ══════════════════════════════════════════════════════════════════════
#  PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════
def run_pipeline(agents, patient_data: dict, patient_id: str) -> dict:
    """Run the full 5-agent pipeline and return shared state."""
    state = {
        "patient_data": patient_data.copy(),
        "patient_id": patient_id,
        "risk": {}, "explanation": "", "recommendation": "",
        "orchestrator_reasoning": "", "updated_recommendation": "",
        "alerts": [], "escalate": False,
        "timestamp": datetime.now().isoformat(),
    }

    # Step 1 — Agent A: Ingestion (validate input)
    cleaned = {k: v for k, v in patient_data.items()}
    state["patient_data"] = cleaned

    # Derive BMI_Category & Glucose_Category if missing
    if "BMI_Category" not in cleaned:
        bmi = cleaned.get("BMI", 0)
        if bmi < 25:
            cleaned["BMI_Category"] = "Normal"
        elif bmi < 30:
            cleaned["BMI_Category"] = "Overweight"
        else:
            cleaned["BMI_Category"] = "Obese"
    if "Glucose_Category" not in cleaned:
        gl = cleaned.get("Glucose", 0)
        if gl < 140:
            cleaned["Glucose_Category"] = "Normal"
        elif gl < 200:
            cleaned["Glucose_Category"] = "Prediabetic"
        else:
            cleaned["Glucose_Category"] = "Diabetic"

    # Step 2 — Agent B: Risk prediction
    prob, risk_level = agents["risk"].predict(cleaned)
    state["risk"] = {"probability": round(prob, 4), "risk_level": risk_level}

    # Step 3 — Agent C: Explainability
    explain_result = agents["explain"].explain(cleaned)
    state["explanation"] = explain_result["explanation_text"]
    state["shap_values"] = explain_result.get("shap_values", {})
    state["top_positive_features"] = explain_result.get("top_positive_features", [])
    state["top_negative_features"] = explain_result.get("top_negative_features", [])

    # Step 4 — Agent D: Recommendation (LLM with fallback)
    if agents["recommend"].llm_available:
        rec = agents["recommend"].generate(
            patient_data=cleaned,
            risk_level=risk_level,
            explanation=state["explanation"],
        )
    else:
        rec = "(LLM unavailable — using template recommendations)"
        result = agents["recommend"].run_pipeline(
            risk_category=risk_level.lower().replace(" ", "_"),
            risk_score=prob, sample_features=cleaned,
            age=cleaned.get("Age", 30), explanation=state["explanation"],
        )
        if result.get("recommendations"):
            r = result["recommendations"][0]
            parts = []
            for cat in ("dietary_recommendations", "exercise_recommendations", "monitoring_plan"):
                items = r.get(cat, [])
                if items:
                    label = cat.replace("_", " ").title()
                    parts.append(f"**{label}:**")
                    parts.extend(f"• {i}" for i in items)
            rec = "\n".join(parts) if parts else rec
    state["recommendation"] = rec

    # Step 5 — Agent E: Monitoring & Trend Detection
    agents["monitor"].update(cleaned, risk_level, patient_id)
    trend = agents["monitor"].analyze_trend(patient_id)
    alerts = agents["monitor"].generate_alerts(patient_id)
    state["trend"] = trend.get("overall", "first_visit")
    state["alerts"] = alerts

    # Step 6 — Agent E: Orchestrator LLM reasoning
    orch = agents["monitor"].orchestrate(state)
    state["orchestrator_reasoning"] = orch["reasoning"]
    state["updated_recommendation"] = orch["updated_recommendation"]
    state["escalate"] = orch["escalate"]

    return state


# ══════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════
def risk_badge(level: str) -> str:
    cls = level.lower().replace(" ", "")
    return f'<span class="risk-badge risk-{cls}">{level}</span>'


def parse_recommendation_sections(text: str) -> dict:
    """Parse recommendation text into Diet/Exercise/Lifestyle/Warning sections."""
    sections = {"Diet": [], "Exercise": [], "Lifestyle": [], "Warning": []}
    current = None
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower().rstrip(":").strip("* ")
        if lower in ("diet", "**diet:**", "**diet**"):
            current = "Diet"; continue
        elif lower in ("exercise", "**exercise:**", "**exercise**"):
            current = "Exercise"; continue
        elif lower in ("lifestyle", "**lifestyle:**", "**lifestyle**"):
            current = "Lifestyle"; continue
        elif lower in ("warning", "**warning:**", "**warning**"):
            current = "Warning"; continue
        if current and stripped:
            clean = stripped.lstrip("•*- ").strip()
            if clean:
                sections[current].append(clean)
    return sections


def render_recommendation_card(rec_text: str):
    """Render recommendations with nested sub-boxes for each category."""
    sections = parse_recommendation_sections(rec_text)
    st.markdown('<div class="glass-card" id="recommendations">', unsafe_allow_html=True)
    st.markdown("#### Recommendations  --  Agent D")
    has_sections = any(v for v in sections.values())
    if has_sections:
        for key in ("Diet", "Exercise", "Lifestyle", "Warning"):
            items = sections.get(key, [])
            if items:
                bullets = "".join(f"<li>{item}</li>" for item in items)
                st.markdown(
                    f'<div class="glass-sub"><h5>{key}</h5>'
                    f'<ul style="margin:0;padding-left:18px">{bullets}</ul></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(rec_text)
    st.markdown("</div>", unsafe_allow_html=True)


def show_disclaimer():
    st.markdown(
        '<div class="disclaimer-box"><b>Medical Disclaimer:</b> '
        "This AI system is for <b>decision support only</b> and is NOT a medical "
        "diagnosis. Always consult a qualified healthcare provider before taking "
        "medical action.</div>",
        unsafe_allow_html=True,
    )


import re
def strip_emojis(text: str) -> str:
    """Remove emoji characters from stored record text."""
    return re.sub(
        r'[\U0001F300-\U0001F9FF\U00002702-\U000027B0\U0000FE00-\U0000FE0F'
        r'\U0000200D\U00002600-\U000026FF\U00002700-\U000027BF]+',
        '', text
    ).strip()


def _clean_text_for_pdf(text: str) -> str:
    """Sanitize text for FPDF: strip emojis and replace unsupported chars."""
    text = strip_emojis(text)
    replacements = {
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--", "\u2026": "...", "\u2022": "*",
        "\u00b7": "*", "\u25cf": "*", "\u2023": ">",
        "\u2192": "->", "\u2190": "<-",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    # Fallback: replace any remaining non-latin1 chars
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text


def generate_pdf_report(patient_id: str, record: dict) -> bytes:
    """Generate a professional PDF report from a patient record dict."""
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Header ──
    pdf.set_fill_color(30, 27, 75)  # Dark indigo
    pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_y(10)
    pdf.cell(0, 10, 'DiabetesAI - Agentic Risk Report',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 7, f'Patient: {patient_id}  |  Date: {record.get("date", "N/A")}',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(12)

    # ── Helpers ──
    def section_title(title: str):
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(99, 102, 241)  # Indigo accent
        pdf.cell(0, 9, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(99, 102, 241)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 180, pdf.get_y())
        pdf.ln(3)
        pdf.set_text_color(40, 40, 40)

    def body_text(text: str):
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(50, 50, 50)
        cleaned = _clean_text_for_pdf(text)
        for line in cleaned.split('\n'):
            line = line.strip()
            if line:
                pdf.multi_cell(0, 5.5, line)
                pdf.ln(1)

    # ── Patient Vitals ──
    section_title('Patient Vitals')
    inp = record.get('input', {})
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    vitals = [
        ('Age', f"{inp.get('Age', '---')} yrs"),
        ('Glucose', f"{inp.get('Glucose', '---')} mg/dL"),
        ('Blood Pressure', f"{inp.get('BloodPressure', '---')} mmHg"),
        ('BMI', f"{inp.get('BMI', '---')}"),
        ('Insulin', f"{inp.get('Insulin', '---')} uU/mL"),
        ('Skin Thickness', f"{inp.get('SkinThickness', '---')} mm"),
        ('Pregnancies', f"{inp.get('Pregnancies', '---')}"),
        ('Diabetes Pedigree Fn', f"{inp.get('DiabetesPedigreeFunction', '---')}"),
    ]
    col_w = 95
    for i in range(0, len(vitals), 2):
        left = vitals[i]
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(35, 6, f"{left[0]}:")
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(col_w - 35, 6, left[1])
        if i + 1 < len(vitals):
            right = vitals[i + 1]
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(35, 6, f"{right[0]}:")
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(col_w - 35, 6, right[1],
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.ln()
    pdf.ln(6)

    # ── Risk Assessment ──
    section_title('Risk Assessment (Agent B)')
    rl = record.get('risk_level', 'Unknown')
    prob = record.get('probability', 0)
    risk_colors = {'Low': (34, 197, 94), 'Moderate': (234, 179, 8),
                   'High': (249, 115, 22), 'Very High': (239, 68, 68)}
    rc = risk_colors.get(rl, (150, 150, 150))
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(*rc)
    pdf.cell(0, 7, f"Risk Level: {rl}  ({prob:.1%} probability)",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(50, 50, 50)
    pdf.ln(4)

    # ── Explanation ──
    section_title('Explanation (Agent C - SHAP)')
    body_text(record.get('explanation', 'N/A'))
    pdf.ln(3)

    # ── Recommendation ──
    section_title('Recommendation (Agent D)')
    body_text(record.get('recommendation', 'N/A'))
    pdf.ln(3)

    # ── Orchestrator Reasoning ──
    orch = record.get('orchestrator_reasoning', '')
    if orch:
        section_title('Orchestrator Reasoning (Agent E)')
        body_text(orch)
        pdf.ln(3)

    # ── Alerts ──
    alerts = record.get('alerts', [])
    if alerts:
        section_title('Monitoring Alerts')
        for a in alerts:
            sev = a.get('severity', 'info').upper()
            msg = _clean_text_for_pdf(a.get('message', ''))
            pdf.set_font('Helvetica', 'B', 10)
            sev_colors = {'CRITICAL': (239, 68, 68), 'HIGH': (249, 115, 22),
                          'MEDIUM': (234, 179, 8)}
            sc = sev_colors.get(sev, (150, 150, 150))
            pdf.set_text_color(*sc)
            pdf.cell(22, 6, f"[{sev}]")
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, msg)
            pdf.ln(1)
        pdf.ln(3)

    # ── Footer / Disclaimer ──
    pdf.ln(5)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 4,
        'Medical Disclaimer: This AI-generated report is for decision support only '
        'and is NOT a medical diagnosis. Always consult a qualified healthcare '
        'provider before taking medical action.'
    )
    pdf.ln(2)
    pdf.set_font('Helvetica', '', 7)
    pdf.cell(0, 4, f'Generated by DiabetesAI Agentic Risk System  |  {record.get("date", "")}', align='C')

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════
def main():
    agents = init_agents()

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Patient Login")
        patient_id = st.text_input(
            "Patient ID", value="", placeholder="e.g. PAT-001",
            help="Enter an existing ID to load history, or a new ID to create a record.",
        )
        if patient_id:
            st.session_state["patient_id"] = patient_id.strip()

        st.markdown("---")
        st.markdown("### Agent Status")
        d_ok = agents['recommend'].llm_available
        st.markdown(
            '<span class="agent-dot on"></span> Agent A -- Ingestion<br>'
            '<span class="agent-dot on"></span> Agent B -- Risk (ML)<br>'
            '<span class="agent-dot on"></span> Agent C -- Explainability<br>'
            f'<span class="agent-dot {"on" if d_ok else "warn"}"></span> Agent D -- Recommendation<br>'
            '<span class="agent-dot on"></span> Agent E -- Orchestrator',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        show_disclaimer()

    pid = st.session_state.get("patient_id", "")

    # ── Header ────────────────────────────────────────────────────────
    st.markdown("# Agentic AI  --  Diabetes Risk System")
    st.caption("Multi-agent pipeline: Ingestion > Risk (ML) > Explainability (SHAP) > Recommendation (LLM) > Orchestrator (LLM)")

    if not pid:
        st.info("Enter a Patient ID in the sidebar to begin.")
        return

    st.markdown(
        f'<div class="glass-card" style="padding:10px 20px;margin-bottom:12px">'
        f'Logged in as <b>{pid}</b></div>',
        unsafe_allow_html=True,
    )

    tab_new, tab_history = st.tabs(["New Assessment", "Patient History"])

    # ── TAB 1: New Assessment ─────────────────────────────────────────
    with tab_new:
        st.markdown("### Enter Patient Vitals")

        with st.form("patient_form", clear_on_submit=False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                pregnancies = st.number_input("Pregnancies", 0, 20, 1)
                glucose = st.number_input("Glucose (mg/dL)", 0.0, 300.0, 120.0, step=1.0)
            with c2:
                bp = st.number_input("Blood Pressure (mmHg)", 0.0, 200.0, 72.0, step=1.0)
                skin = st.number_input("Skin Thickness (mm)", 0.0, 100.0, 29.0, step=1.0)
            with c3:
                insulin = st.number_input("Insulin (uU/mL)", 0.0, 900.0, 100.0, step=1.0)
                bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
            with c4:
                dpf = st.number_input("Diabetes Pedigree Fn", 0.0, 2.5, 0.35, step=0.01)
                age = st.number_input("Age (years)", 1, 120, 30)

            submitted = st.form_submit_button(
                "Run Full Agentic Pipeline", width='stretch', type="primary",
            )

        if submitted:
            if glucose <= 0 or bmi <= 0:
                st.error("Glucose and BMI must be positive values.")
                return

            patient_data = {
                "Pregnancies": int(pregnancies),
                "Glucose": float(glucose),
                "BloodPressure": float(bp),
                "SkinThickness": float(skin),
                "Insulin": float(insulin),
                "BMI": float(bmi),
                "DiabetesPedigreeFunction": float(dpf),
                "Age": int(age),
            }

            with st.spinner("Running 5-agent pipeline ..."):
                try:
                    state = run_pipeline(agents, patient_data, pid)
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    return

            st.session_state["latest_state"] = state

            # Save record
            record = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "input": patient_data,
                "risk_level": state["risk"]["risk_level"],
                "probability": state["risk"]["probability"],
                "explanation": state["explanation"],
                "recommendation": state["recommendation"],
                "orchestrator_reasoning": state["orchestrator_reasoning"],
                "alerts": [
                    {"severity": a["severity"], "message": a["message"]}
                    for a in state.get("alerts", [])
                ],
                "escalate": state["escalate"],
            }
            save_record(pid, record)

            # ── Display results ───────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("## Pipeline Results")

            # 1. Patient Summary
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Patient Summary")
            ps1, ps2, ps3, ps4 = st.columns(4)
            ps1.metric("Age", f"{patient_data['Age']} yrs")
            ps2.metric("Glucose", f"{patient_data['Glucose']} mg/dL")
            ps3.metric("BMI", f"{patient_data['BMI']}")
            ps4.metric("Blood Pressure", f"{patient_data['BloodPressure']} mmHg")
            st.markdown("</div>", unsafe_allow_html=True)

            # 2. Risk Prediction
            rl = state["risk"]["risk_level"]
            prob = state["risk"]["probability"]
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Risk Prediction  --  Agent B")
            r1, r2, r3 = st.columns([2, 2, 3])
            with r1:
                st.metric("Risk Probability", f"{prob:.1%}")
            with r2:
                st.markdown(f"**Risk Level:** {risk_badge(rl)}", unsafe_allow_html=True)
            with r3:
                st.progress(min(prob, 1.0))
            st.markdown("</div>", unsafe_allow_html=True)

            if rl in ("High", "Very High"):
                st.markdown(
                    '<div class="alert-critical"><b>HIGH RISK DETECTED</b> -- '
                    'Please consult a doctor immediately.</div>',
                    unsafe_allow_html=True,
                )

            # 3. Explanation
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Explanation  --  Agent C (SHAP)")
            st.markdown(
                f'<div class="glass-sub">{state["explanation"]}</div>',
                unsafe_allow_html=True,
            )
            top_feats = state.get("top_positive_features", [])
            if top_feats:
                st.markdown("**Top Risk-Increasing Features:**")
                bar_color = RISK_COLORS.get(rl, "#818cf8")
                for fname, val in top_feats[:5]:
                    bar_w = min(int(abs(val) * 800), 200)
                    val_str = f"{val:+.4f}"
                    st.markdown(
                        f"<span style='display:inline-block;width:220px;color:#e2e8f0'>{fname}</span>"
                        f"<span style='display:inline-block;width:{bar_w}px;height:12px;"
                        f"background:{bar_color};border-radius:4px'></span>"
                        f" <code style='color:#94a3b8'>{val_str}</code>",
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

            # 4. Recommendations (with sub-boxes)
            render_recommendation_card(state["recommendation"])

            # 5. Orchestrator Reasoning
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Orchestrator Reasoning  --  Agent E")
            st.markdown(
                f'<div class="glass-sub">{state["orchestrator_reasoning"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # 6. Monitoring / Alerts
            if state["alerts"]:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Monitoring / Alerts")
                for a in state["alerts"]:
                    sev = a["severity"]
                    cls = "alert-critical" if sev == "critical" else (
                        "alert-high" if sev == "high" else "alert-medium")
                    st.markdown(
                        f'<div class="{cls}"><b>[{sev.upper()}]</b> {a["message"]}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            show_disclaimer()

            # ── PDF Download Button ──────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            pdf_record = {
                "date": record["date"],
                "input": patient_data,
                "risk_level": state["risk"]["risk_level"],
                "probability": state["risk"]["probability"],
                "explanation": state["explanation"],
                "recommendation": state["recommendation"],
                "orchestrator_reasoning": state["orchestrator_reasoning"],
                "alerts": [
                    {"severity": a["severity"], "message": a["message"]}
                    for a in state.get("alerts", [])
                ],
            }
            pdf_bytes = generate_pdf_report(pid, pdf_record)
            ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Report as PDF",
                data=pdf_bytes,
                file_name=f"{pid}_report_{ts_slug}.pdf",
                mime="application/pdf",
                width='stretch',
            )

    # ── TAB 2: Patient History ────────────────────────────────────────
    with tab_history:
        records = load_patient(pid)
        if not records:
            st.info("No previous records found for this patient.")
        else:
            st.markdown(f"### History for **{pid}** ({len(records)} records)")
            
            # --- Longitudinal Graph (Agent E Monitoring) — Interactive ---
            if len(records) > 1:
                import plotly.graph_objects as go

                # Sort records chronologically for the graph
                sorted_records = sorted(records, key=lambda r: r.get('date', ''))
                dates = [r.get('date', '').split(' ')[0] for r in sorted_records]

                # If multiple records have the same date, append time to make them distinct
                if len(set(dates)) < len(dates):
                    dates = [r.get('date', '')[5:16] for r in sorted_records]  # 'MM-DD HH:MM'

                glucose_vals = [float(r.get('input', {}).get('Glucose', 0)) for r in sorted_records]
                risk_vals = [float(r.get('probability', 0)) * 100 for r in sorted_records]
                risk_levels = [r.get('risk_level', 'N/A') for r in sorted_records]

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Longitudinal Trend (Agent E)")

                fig = go.Figure()

                # Glucose trace (primary y-axis)
                fig.add_trace(go.Scatter(
                    x=dates, y=glucose_vals,
                    name='Glucose',
                    mode='lines+markers',
                    marker=dict(size=10, color='#818cf8', symbol='circle'),
                    line=dict(width=3, color='#818cf8'),
                    hovertemplate=(
                        '<b>%{x}</b><br>'
                        'Glucose: <b>%{y:.1f} mg/dL</b>'
                        '<extra></extra>'
                    ),
                    yaxis='y1',
                ))

                # Risk % trace (secondary y-axis)
                fig.add_trace(go.Scatter(
                    x=dates, y=risk_vals,
                    name='Risk %',
                    mode='lines+markers',
                    marker=dict(size=10, color='#ef4444', symbol='square'),
                    line=dict(width=3, color='#ef4444'),
                    customdata=risk_levels,
                    hovertemplate=(
                        '<b>%{x}</b><br>'
                        'Risk: <b>%{y:.1f}%</b> (%{customdata})'
                        '<extra></extra>'
                    ),
                    yaxis='y2',
                ))

                # Glucose threshold lines
                fig.add_hline(
                    y=140, line_dash='dash', line_color='#eab308',
                    opacity=0.6, annotation_text='Warning (140)',
                    annotation_position='top left',
                    annotation_font_color='#eab308',
                )
                fig.add_hline(
                    y=180, line_dash='dash', line_color='#ef4444',
                    opacity=0.6, annotation_text='Critical (180)',
                    annotation_position='top left',
                    annotation_font_color='#ef4444',
                )

                # Layout with dual y-axes and dark transparent theme
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    margin=dict(l=50, r=50, t=30, b=50),
                    legend=dict(
                        orientation='h', yanchor='bottom', y=1.02,
                        xanchor='left', x=0,
                        font=dict(color='#e2e8f0'),
                    ),
                    hovermode='x unified',
                    yaxis=dict(
                        title=dict(text='Glucose (mg/dL)', font=dict(color='#818cf8')),
                        tickfont=dict(color='#818cf8'),
                        gridcolor='rgba(255,255,255,0.08)',
                        range=[
                            min(80, min(glucose_vals) - 20),
                            max(220, max(glucose_vals) + 20),
                        ],
                    ),
                    yaxis2=dict(
                        title=dict(text='Risk Probability (%)', font=dict(color='#ef4444')),
                        tickfont=dict(color='#ef4444'),
                        overlaying='y', side='right',
                        range=[0, 100],
                        showgrid=False,
                    ),
                    xaxis=dict(
                        tickfont=dict(color='#e2e8f0'),
                        gridcolor='rgba(255,255,255,0.08)',
                    ),
                )

                st.plotly_chart(fig, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
            # -----------------------------------------------

            for i, rec in enumerate(reversed(records)):
                rl = rec.get("risk_level", "Unknown")
                with st.expander(
                    f"{rec.get('date', 'N/A')}  |  {rl}  "
                    f"(prob {rec.get('probability', 0):.1%})",
                    expanded=(i == 0),
                ):
                    mc1, mc2, mc3 = st.columns(3)
                    inp = rec.get("input", {})
                    mc1.metric("Glucose", f"{inp.get('Glucose', '---')} mg/dL")
                    mc2.metric("BMI", f"{inp.get('BMI', '---')}")
                    mc3.metric("Age", f"{inp.get('Age', '---')}")

                    st.markdown("**Explanation:**")
                    st.markdown(strip_emojis(rec.get("explanation", "N/A")))

                    st.markdown("**Recommendation:**")
                    st.markdown(strip_emojis(rec.get("recommendation", "N/A")))

                    if rec.get("orchestrator_reasoning"):
                        st.markdown("**Orchestrator Reasoning:**")
                        st.markdown(strip_emojis(rec.get("orchestrator_reasoning", "N/A")))

                    if rec.get("alerts"):
                        st.markdown("**Alerts:**")
                        for a in rec["alerts"]:
                            st.warning(f"[{a['severity'].upper()}] {a['message']}")

                    # ── PDF download for this history record ──
                    hist_pdf = generate_pdf_report(pid, rec)
                    date_slug = rec.get('date', 'unknown').replace(' ', '_').replace(':', '')
                    st.download_button(
                        label="Download This Report as PDF",
                        data=hist_pdf,
                        file_name=f"{pid}_report_{date_slug}.pdf",
                        mime="application/pdf",
                        key=f"dl_hist_{i}_{date_slug}",
                        width='stretch',
                    )


if __name__ == "__main__":
    main()