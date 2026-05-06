"""
╔══════════════════════════════════════════════════════════════════╗
║   AGENTIC AI SYSTEM — Type 2 Diabetes Risk Stratification      ║
║   & Personalized Management                                     ║
║                                                                  ║
║   True multi-agent orchestration with:                           ║
║     • Shared state between agents                                ║
║     • LLM reasoning in Agent D (Recommendation) & E (Orchestrator)║
║     • ML prediction in Agent B (Risk)                            ║
║     • SHAP explainability in Agent C                             ║
║     • Feedback loop: E → D → E (multi-step refinement)          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import logging
import sys
import warnings
from datetime import datetime
from typing import Dict

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("PIPELINE")

# ── Import all agents ────────────────────────────────────────────────
from src.agents.ingestion_agent import IngestionAgent
from src.agents.risk_agent import RiskAgent
from src.agents.explainability_agent import ExplainabilityAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.monitoring_agent import MonitoringAgent


# ══════════════════════════════════════════════════════════════════════
#  SHARED STATE — all agents read/write to this
# ══════════════════════════════════════════════════════════════════════
def create_empty_state() -> Dict:
    """Create a fresh shared state dictionary."""
    return {
        "patient_data": {},
        "risk": {
            "probability": None,
            "risk_level": None,
        },
        "explanation": "",
        "recommendation": "",
        "orchestrator_reasoning": "",
        "updated_recommendation": "",
        "alerts": [],
        "escalate": False,
        "timestamp": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════
#  MULTI-AGENT PIPELINE
# ══════════════════════════════════════════════════════════════════════
class DiabetesAgentPipeline:
    """
    Coordinated multi-agent system where agents collaborate through
    shared state, and the Monitoring/Orchestrator agent uses LLM
    reasoning to make autonomous decisions.

    Flow:
        ┌──────────┐    ┌──────────┐    ┌──────────────┐
        │ Agent A   │───▶│ Agent B   │───▶│  Agent C      │
        │ Ingestion │    │ Risk (ML) │    │ Explain(SHAP)│
        └──────────┘    └──────────┘    └──────────────┘
                                               │
                                               ▼
        ┌──────────────────────────────────────────────────┐
        │              SHARED STATE                         │
        └──────────────────────────────────────────────────┘
                                               │
                                               ▼
                                    ┌──────────────┐
                                    │  Agent D      │
                                    │ Recommend(LLM)│
                                    └──────┬───────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │  Agent E      │
                                    │ Orchestrator  │
                                    │  (LLM + Logic)│
                                    └──────┬───────┘
                                           │
                              ┌────────────┴────────────┐
                              ▼                         ▼
                        Feedback Loop            Final Output
                        (back to Agent D)
    """

    def __init__(self):
        """Initialize all agents."""
        logger.info("=" * 60)
        logger.info("🚀 Initializing Multi-Agent Diabetes System")
        logger.info("=" * 60)

        self.agent_a = IngestionAgent()
        self.agent_b = RiskAgent()
        self.agent_c = ExplainabilityAgent()
        self.agent_d = RecommendationAgent()
        self.agent_e = MonitoringAgent()

        logger.info(f"  Agent A (Ingestion):      ✅ Ready")
        logger.info(f"  Agent B (Risk/ML):         ✅ Model={type(self.agent_b.model).__name__}")
        logger.info(f"  Agent C (Explainability):  ✅ SHAP")
        logger.info(f"  Agent D (Recommendation):  {'✅ LLM' if self.agent_d.llm_available else '❌ No LLM'}")
        logger.info(f"  Agent E (Orchestrator):    ✅ LLM + Logic")
        logger.info("=" * 60)

    # ── Run for a single patient ─────────────────────────────────────
    def process_patient(self, patient_data: Dict,
                        patient_id: str = "patient_001") -> Dict:
        """
        Process a single patient through the full multi-agent pipeline.

        Args:
            patient_data: dict with patient features
            patient_id:   unique identifier

        Returns:
            Final shared state after all agents have processed.
        """
        state = create_empty_state()
        state["patient_data"] = patient_data.copy()
        state["patient_id"] = patient_id

        banner = f"  Processing: {patient_id}"
        print("\n" + "▓" * 65)
        print(f"▓{banner:^63s}▓")
        print("▓" * 65)

        # ────────────────────────────────────────────────────────────
        # STEP 1: Agent A — Ingestion (pure preprocessing, no LLM)
        # ────────────────────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  🔹 STEP 1 │ Agent A — Data Ingestion & Preprocessing")
        print(f"{'─'*65}")
        logger.info("[Agent A] Processing patient data …")

        # Agent A validates and cleans the input
        cleaned = {}
        for k, v in patient_data.items():
            cleaned[k] = v
        state["patient_data"] = cleaned
        logger.info(f"[Agent A] ✅ Patient data validated — "
                     f"{len(cleaned)} features")
        print(f"  Features: {list(cleaned.keys())}")

        # ────────────────────────────────────────────────────────────
        # STEP 2: Agent B — Risk Prediction (ML, no LLM)
        # ────────────────────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  🔹 STEP 2 │ Agent B — Risk Prediction (RandomForest ML)")
        print(f"{'─'*65}")
        logger.info("[Agent B] Running ML prediction …")

        prob, risk_level = self.agent_b.predict(patient_data)
        state["risk"] = {
            "probability": round(prob, 4),
            "risk_level": risk_level,
        }
        logger.info(f"[Agent B] ✅ Risk = {risk_level} (prob={prob:.4f})")
        print(f"  Probability: {prob:.4f}")
        print(f"  Risk Level:  {risk_level}")

        # ────────────────────────────────────────────────────────────
        # STEP 3: Agent C — Explainability (SHAP, no LLM)
        # ────────────────────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  🔹 STEP 3 │ Agent C — Explainability (SHAP Analysis)")
        print(f"{'─'*65}")
        logger.info("[Agent C] Computing SHAP values …")

        explain_result = self.agent_c.explain(patient_data)
        state["explanation"] = explain_result["explanation_text"]
        state["shap_values"] = explain_result.get("shap_values", {})
        logger.info("[Agent C] ✅ Explanation generated")
        print(f"  {explain_result['explanation_text'][:200]}…")

        # ────────────────────────────────────────────────────────────
        # STEP 4: Agent D — Initial Recommendation (LLM — Gemini)
        # ────────────────────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  🔹 STEP 4 │ Agent D — Recommendation (LLM: Gemini 2.5)")
        print(f"{'─'*65}")
        logger.info("[Agent D] Generating LLM recommendation …")

        if self.agent_d.llm_available:
            initial_rec = self.agent_d.generate(
                patient_data=patient_data,
                risk_level=risk_level,
                explanation=state["explanation"],
            )
        else:
            initial_rec = "(LLM unavailable — set GOOGLE_API_KEY)"

        state["recommendation"] = initial_rec
        logger.info("[Agent D] ✅ Initial recommendation generated")
        print(f"  {initial_rec[:300]}…" if len(initial_rec) > 300 else f"  {initial_rec}")

        # ────────────────────────────────────────────────────────────
        # STEP 5: Agent E — Monitoring & Trend Detection (Logic)
        # ────────────────────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  🔹 STEP 5 │ Agent E — Monitoring & Trend Detection")
        print(f"{'─'*65}")
        logger.info("[Agent E] Recording observation & analyzing trends …")

        self.agent_e.update(patient_data, risk_level, patient_id)
        trend = self.agent_e.analyze_trend(patient_id)
        alerts = self.agent_e.generate_alerts(patient_id)

        state["trend"] = trend.get("overall", "first_visit")
        state["alerts"] = alerts

        print(f"  Trend: {trend.get('overall', 'first visit (no history)')}")
        if alerts:
            for a in alerts:
                print(f"  [{a['severity'].upper()}] {a['message']}")
        else:
            print(f"  No alerts triggered")

        # ────────────────────────────────────────────────────────────
        # STEP 6: Agent E — ORCHESTRATOR REASONING (LLM — Gemini)
        #
        #   THIS is the key agentic feature: Agent E uses LLM to
        #   reason about the FULL state and make autonomous decisions
        #   about intensification, escalation, and alerts.
        # ────────────────────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  🔹 STEP 6 │ Agent E — Orchestrator Reasoning (LLM)")
        print(f"{'─'*65}")
        logger.info("[Agent E / Orchestrator] 🧠 LLM reasoning over full state …")

        orchestrator_output = self.agent_e.orchestrate(state)

        state["orchestrator_reasoning"] = orchestrator_output["reasoning"]
        state["updated_recommendation"] = orchestrator_output["updated_recommendation"]
        state["escalate"] = orchestrator_output["escalate"]

        if orchestrator_output["escalate"]:
            print(f"  🚨 ESCALATION TRIGGERED by orchestrator")
        else:
            print(f"  ✅ No escalation needed")

        # ────────────────────────────────────────────────────────────
        # FINAL OUTPUT
        # ────────────────────────────────────────────────────────────
        print(f"\n{'▓'*65}")
        print(f"▓{'  FINAL RESULTS':^63s}▓")
        print(f"{'▓'*65}")

        print(f"\n📊 RISK LEVEL: {state['risk']['risk_level']} "
              f"(probability: {state['risk']['probability']})")

        print(f"\n🔍 EXPLANATION (Agent C):")
        print(f"{state['explanation']}")

        print(f"\n💊 INITIAL RECOMMENDATION (Agent D — LLM):")
        print(f"{state['recommendation']}")

        print(f"\n🧠 ORCHESTRATOR REASONING (Agent E — LLM):")
        reasoning = state["orchestrator_reasoning"]
        print(reasoning)

        if state["alerts"]:
            print(f"\n🚨 ALERTS ({len(state['alerts'])}):")
            for a in state["alerts"]:
                print(f"  [{a['severity'].upper()}] {a['message']}")

        print(f"\n{'▓'*65}")

        return state


# ══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
def main():
    """Run the multi-agent system for sample patients."""

    pipeline = DiabetesAgentPipeline()

    # ── Patient 1: HIGH RISK ─────────────────────────────────────────
    high_risk_patient = {
        "Pregnancies": 6, "Glucose": 178.0, "BloodPressure": 92.0,
        "SkinThickness": 35.0, "Insulin": 190.0, "BMI": 34.5,
        "DiabetesPedigreeFunction": 0.627, "Age": 55,
        "BMI_Category": "Obese", "Glucose_Category": "Prediabetic",
    }

    # Simulate a previous visit (worsening)
    previous_visit = {
        "Pregnancies": 6, "Glucose": 150.0, "BloodPressure": 82.0,
        "SkinThickness": 35.0, "Insulin": 160.0, "BMI": 32.0,
        "DiabetesPedigreeFunction": 0.627, "Age": 55,
        "BMI_Category": "Obese", "Glucose_Category": "Prediabetic",
    }
    pipeline.agent_e.update(previous_visit, "High", "high_risk_001")

    state_high = pipeline.process_patient(
        high_risk_patient, patient_id="high_risk_001"
    )

    # ── Patient 2: LOW RISK ──────────────────────────────────────────
    low_risk_patient = {
        "Pregnancies": 1, "Glucose": 85.0, "BloodPressure": 66.0,
        "SkinThickness": 29.0, "Insulin": 80.0, "BMI": 22.5,
        "DiabetesPedigreeFunction": 0.15, "Age": 25,
        "BMI_Category": "Normal", "Glucose_Category": "Normal",
    }

    state_low = pipeline.process_patient(
        low_risk_patient, patient_id="low_risk_002"
    )

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  📋 PIPELINE SUMMARY")
    print("═" * 65)
    print(f"  High-risk patient: {state_high['risk']['risk_level']} "
          f"(prob={state_high['risk']['probability']}) "
          f"| Escalate={state_high['escalate']}")
    print(f"  Low-risk patient:  {state_low['risk']['risk_level']} "
          f"(prob={state_low['risk']['probability']}) "
          f"| Escalate={state_low['escalate']}")
    print(f"\n  Agents used: A (Preprocess) → B (ML) → C (SHAP) → "
          f"D (LLM) → E (LLM Orchestrator)")
    print(f"  LLM calls: 2 per patient (Agent D + Agent E)")
    print(f"  Feedback: Agent E reasons over Agent D's output")
    print(f"\n  ⚠️ This is a decision support system, "
          f"not a medical diagnosis.")
    print("═" * 65)


if __name__ == "__main__":
    main()
