"""
Monitoring & Follow-up Orchestrator Agent (Agent E).

Central coordinating agent that:
  1. Tracks patient health metrics over time (stateful)
  2. Detects worsening / improving trends
  3. Triggers alerts when thresholds are crossed
  4. Adapts recommendations dynamically via feedback loop to Agent D
  5. Simulates a real-world longitudinal monitoring system

Pipeline position:
  Ingestion → Risk → Explain → Recommend → **Monitoring → Feedback**

⚠️ This system supports monitoring but does NOT replace medical supervision.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

from src.config.settings import (
    MONITORING_WINDOW_MONTHS, ALERT_THRESHOLD, MEDICAL_DISCLAIMER
)

logger = logging.getLogger(__name__)

# ── Clinical thresholds ──────────────────────────────────────────────
THRESHOLDS = {
    "Glucose":       {"warning": 140, "critical": 180, "unit": "mg/dL"},
    "BMI":           {"warning": 30,  "critical": 35,  "unit": ""},
    "BloodPressure": {"warning": 90,  "critical": 100, "unit": "mmHg"},
    "Insulin":       {"warning": 166, "critical": 250, "unit": "μU/mL"},
}

RISK_ORDER = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3}


class MonitoringAgent:
    """
    Agent E — Stateful monitoring and feedback-loop orchestrator.

    Unlike previous agents (stateless), this one maintains patient history
    and triggers adaptive responses across the agent system.

    Usage:
        agent = MonitoringAgent()
        agent.update(patient_data, risk_level)
        agent.update(patient_data_v2, new_risk_level)
        trend   = agent.analyze_trend()
        alerts  = agent.generate_alerts()
        adapted = agent.adjust_recommendation(base_recommendation)
    """

    # ── __init__ ─────────────────────────────────────────────────────
    def __init__(self):
        """Initialize with empty patient history."""
        self.history: List[Dict] = []          # time-ordered records
        self.patient_history: Dict[str, List[Dict]] = {}  # per-patient
        self.alerts: List[Dict] = []
        self.trend_analysis: Dict = {}

    # ── 1. Track Patient Data Over Time ──────────────────────────────
    def update(self, patient_data: Dict, risk_level: str,
               patient_id: str = "default") -> None:
        """
        Record a new observation for a patient.

        Args:
            patient_data: dict with keys like Glucose, BMI, BloodPressure, etc.
            risk_level:   "Low" / "Moderate" / "High" / "Very High"
            patient_id:   unique patient identifier
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_id,
            "risk_level": risk_level,
            **{k: v for k, v in patient_data.items()
               if isinstance(v, (int, float))},
        }

        # Global history
        self.history.append(record)

        # Per-patient history
        if patient_id not in self.patient_history:
            self.patient_history[patient_id] = []
        self.patient_history[patient_id].append(record)

        logger.info(
            f"[MonitoringAgent] Recorded observation #{len(self.history)} "
            f"for {patient_id} — risk={risk_level}"
        )

    # ── 2. Detect Trends ─────────────────────────────────────────────
    def analyze_trend(self, patient_id: str = "default") -> Dict:
        """
        Compare last 2–3 records to detect increasing / decreasing trends.

        Returns:
            Dict with per-metric trend direction and overall assessment.
        """
        records = self.patient_history.get(patient_id, self.history)

        if len(records) < 2:
            return {"status": "insufficient_data",
                    "message": "Need at least 2 observations to detect trends."}

        # Use the last 3 records (or fewer)
        recent = records[-3:]
        first, last = recent[0], recent[-1]

        metric_trends = {}
        tracked = ["Glucose", "BMI", "BloodPressure", "Insulin"]

        for metric in tracked:
            if metric in first and metric in last:
                v_first = float(first[metric])
                v_last = float(last[metric])
                if v_first == 0:
                    continue
                pct_change = ((v_last - v_first) / v_first) * 100

                if pct_change > 3:
                    direction = "worsening"
                elif pct_change < -3:
                    direction = "improving"
                else:
                    direction = "stable"

                metric_trends[metric] = {
                    "first": v_first,
                    "last": v_last,
                    "pct_change": round(pct_change, 2),
                    "direction": direction,
                }

        # Risk-level trend
        risk_first = RISK_ORDER.get(first.get("risk_level", "Low"), 0)
        risk_last = RISK_ORDER.get(last.get("risk_level", "Low"), 0)
        if risk_last > risk_first:
            risk_trend = "worsening"
        elif risk_last < risk_first:
            risk_trend = "improving"
        else:
            risk_trend = "stable"

        # Overall assessment
        worsening_count = sum(
            1 for m in metric_trends.values() if m["direction"] == "worsening"
        )
        improving_count = sum(
            1 for m in metric_trends.values() if m["direction"] == "improving"
        )

        if worsening_count >= 2 or risk_trend == "worsening":
            overall = "worsening"
        elif improving_count >= 2 and risk_trend != "worsening":
            overall = "improving"
        else:
            overall = "stable"

        self.trend_analysis = {
            "status": "success",
            "patient_id": patient_id,
            "observations_analyzed": len(recent),
            "metric_trends": metric_trends,
            "risk_trend": risk_trend,
            "overall": overall,
        }
        return self.trend_analysis

    # ── 3. Generate Alerts ───────────────────────────────────────────
    def generate_alerts(self, patient_id: str = "default") -> List[Dict]:
        """
        Check the latest record and trend analysis for alert conditions.

        Alert rules:
          • risk_level == "Very High"     → 🚨 critical
          • Glucose > 180                → ⚠️ warning
          • Trend worsening              → ⚠️ risk increasing
          • Risk escalated from previous → ⚠️ escalation
        """
        records = self.patient_history.get(patient_id, self.history)
        if not records:
            return []

        latest = records[-1]
        alerts: List[Dict] = []
        now = datetime.now().isoformat()

        # Rule 1: Very High risk
        if latest.get("risk_level") == "Very High":
            alerts.append({
                "severity": "critical",
                "type": "very_high_risk",
                "message": "🚨 Critical: Risk level is VERY HIGH — "
                           "immediate clinical attention required.",
                "timestamp": now,
                "action": "Seek immediate medical consultation.",
            })

        # Rule 2: Glucose above critical threshold
        glucose = latest.get("Glucose", 0)
        if glucose > THRESHOLDS["Glucose"]["critical"]:
            alerts.append({
                "severity": "high",
                "type": "glucose_critical",
                "message": f"⚠️ Glucose is critically elevated at {glucose} mg/dL "
                           f"(threshold: {THRESHOLDS['Glucose']['critical']} mg/dL).",
                "timestamp": now,
                "action": "Monitor glucose closely; consult physician.",
            })
        elif glucose > THRESHOLDS["Glucose"]["warning"]:
            alerts.append({
                "severity": "medium",
                "type": "glucose_warning",
                "message": f"⚠️ Glucose is elevated at {glucose} mg/dL.",
                "timestamp": now,
                "action": "Review dietary plan; increase monitoring frequency.",
            })

        # Rule 3: Risk escalation between visits
        if len(records) >= 2:
            prev_risk = RISK_ORDER.get(records[-2].get("risk_level", "Low"), 0)
            curr_risk = RISK_ORDER.get(latest.get("risk_level", "Low"), 0)
            if curr_risk > prev_risk:
                alerts.append({
                    "severity": "high",
                    "type": "risk_escalation",
                    "message": f"⚠️ Risk escalated from "
                               f"{records[-2].get('risk_level')} → "
                               f"{latest.get('risk_level')}.",
                    "timestamp": now,
                    "action": "Schedule follow-up appointment.",
                })

        # Rule 4: Worsening trend
        trend = self.trend_analysis if self.trend_analysis else self.analyze_trend(patient_id)
        if trend.get("overall") == "worsening":
            alerts.append({
                "severity": "high",
                "type": "trend_worsening",
                "message": "⚠️ Overall health trend is worsening over recent visits.",
                "timestamp": now,
                "action": "Intensify lifestyle interventions; consult physician.",
            })

        # BMI critical
        bmi = latest.get("BMI", 0)
        if bmi > THRESHOLDS["BMI"]["critical"]:
            alerts.append({
                "severity": "medium",
                "type": "bmi_critical",
                "message": f"⚠️ BMI is {bmi} — severe obesity range.",
                "timestamp": now,
                "action": "Refer to dietitian and exercise specialist.",
            })

        self.alerts = alerts
        return alerts

    # ── 4. Adjust Recommendation (Feedback Loop) ─────────────────────
    def adjust_recommendation(self, recommendation: str,
                              patient_id: str = "default") -> str:
        """
        Dynamically adapt a recommendation based on monitoring state.

        This is the **feedback loop**: monitoring results flow back into
        the recommendation layer.

        Args:
            recommendation: base recommendation text from Agent D
            patient_id:     patient identifier

        Returns:
            Adjusted recommendation string with monitoring addendums.
        """
        trend = self.trend_analysis if self.trend_analysis else self.analyze_trend(patient_id)
        alerts = self.alerts if self.alerts else self.generate_alerts(patient_id)

        addendums: List[str] = []

        # Worsening → strengthen
        if trend.get("overall") == "worsening":
            addendums.append(
                "\n📈 MONITORING UPDATE — Trend Worsening:\n"
                "• Increase monitoring frequency to weekly.\n"
                "• Consult physician urgently to review care plan.\n"
                "• Stricter adherence to diet and exercise is critical."
            )
            # Per-metric specifics
            for metric, info in trend.get("metric_trends", {}).items():
                if info["direction"] == "worsening":
                    addendums.append(
                        f"  → {metric} increased by {info['pct_change']:.1f}% "
                        f"({info['first']} → {info['last']})"
                    )

        elif trend.get("overall") == "improving":
            addendums.append(
                "\n📉 MONITORING UPDATE — Trend Improving:\n"
                "• Great progress! Continue current plan.\n"
                "• Maintain regular check-ups to sustain improvement."
            )

        # Critical alerts
        critical = [a for a in alerts if a["severity"] == "critical"]
        if critical:
            addendums.append(
                "\n🚨 CRITICAL ALERT:\n"
                "• " + critical[0]["message"] + "\n"
                "• " + critical[0]["action"]
            )

        # Disclaimer
        addendums.append(
            "\n⚠️ This system supports monitoring but does NOT "
            "replace medical supervision."
        )

        if addendums:
            return recommendation + "\n" + "\n".join(addendums)
        return recommendation

    # ── 5. Feedback Loop: call back into Recommendation Agent ────────
    def feedback_loop(self, recommendation_agent, patient_data: Dict,
                      risk_level: str, explanation: str,
                      patient_id: str = "default") -> str:
        """
        Complete feedback loop: re-invoke the Recommendation Agent (Agent D)
        with updated monitoring context, then adjust the output.

        Args:
            recommendation_agent: instance of RecommendationAgent (Agent D)
            patient_data: current patient features
            risk_level:   current risk level string
            explanation:  explanation from Agent C
            patient_id:   patient identifier

        Returns:
            Final adjusted recommendation string.
        """
        # Analyze current state
        trend = self.analyze_trend(patient_id)
        alerts = self.generate_alerts(patient_id)

        # Enrich explanation with monitoring context
        monitoring_context = f"\n\nMonitoring status: {trend.get('overall', 'unknown')}."
        if trend.get("metric_trends"):
            for m, info in trend["metric_trends"].items():
                monitoring_context += (
                    f"\n  {m}: {info['direction']} "
                    f"({info['pct_change']:+.1f}%)"
                )
        if alerts:
            monitoring_context += f"\nActive alerts: {len(alerts)}"
            for a in alerts:
                monitoring_context += f"\n  - {a['message']}"

        enriched_explanation = explanation + monitoring_context

        # Re-invoke Recommendation Agent with enriched context
        if hasattr(recommendation_agent, "generate") and \
           getattr(recommendation_agent, "llm_available", False):
            logger.info("[MonitoringAgent] Feedback loop → calling Recommendation Agent")
            base_rec = recommendation_agent.generate(
                patient_data=patient_data,
                risk_level=risk_level,
                explanation=enriched_explanation,
            )
        else:
            base_rec = "(Recommendation Agent not available for feedback loop)"

        # Apply monitoring adjustments on top
        return self.adjust_recommendation(base_rec, patient_id)

    # ── Pipeline interface (backward-compatible) ─────────────────────
    def run_pipeline(self, patient_id: str, current_risk_score: float,
                     current_metrics: Dict,
                     previous_risk_score: Optional[float] = None) -> Dict:
        """
        Run the monitoring pipeline (compatible with main_pipeline.py).

        Args:
            patient_id:          unique identifier
            current_risk_score:  probability 0-1
            current_metrics:     patient feature dict
            previous_risk_score: previous probability for comparison

        Returns:
            Monitoring report dict.
        """
        logger.info("=" * 50)
        logger.info("MONITORING AGENT: Starting pipeline")
        logger.info("=" * 50)

        # Map score to risk level
        if current_risk_score >= 0.75:
            risk_level = "Very High"
        elif current_risk_score >= 0.50:
            risk_level = "High"
        elif current_risk_score >= 0.25:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        # If we have a previous score, add a synthetic previous record
        if previous_risk_score is not None and \
           patient_id not in self.patient_history:
            prev_level = "Low"
            if previous_risk_score >= 0.75:   prev_level = "Very High"
            elif previous_risk_score >= 0.50: prev_level = "High"
            elif previous_risk_score >= 0.25: prev_level = "Moderate"
            self.update(current_metrics, prev_level, patient_id)

        # Record current observation
        self.update(current_metrics, risk_level, patient_id)

        # Analyze
        trend = self.analyze_trend(patient_id)
        alerts = self.generate_alerts(patient_id)

        report = {
            "status": "success",
            "patient_id": patient_id,
            "current_risk_score": current_risk_score,
            "current_risk_level": risk_level,
            "previous_risk_score": previous_risk_score,
            "trend_analysis": trend,
            "alerts": alerts,
            "num_alerts": len(alerts),
            "history_length": len(self.patient_history.get(patient_id, [])),
            "recommendation": "Continue monitoring" if not alerts else "Escalate care",
            "disclaimer": "This system supports monitoring but does NOT "
                          "replace medical supervision.",
        }

        logger.info(f"MONITORING AGENT: {len(alerts)} alerts generated, "
                     f"trend={trend.get('overall', 'n/a')}")
        return report

    # ── 6. LLM Orchestrator Reasoning (KEY AGENTIC FEATURE) ─────────
    def orchestrate(self, state: Dict) -> Dict:
        """
        Use LLM to reason about the full system state and make
        autonomous decisions. This is what makes the system truly agentic.

        The LLM analyzes risk, explanation, and initial recommendation,
        then decides:
          1. Should recommendations be intensified?
          2. Should an alert be triggered?
          3. Should the patient be escalated?

        Args:
            state: shared pipeline state dict with keys:
                   patient_data, risk, explanation, recommendation

        Returns:
            Dict with: reasoning, updated_recommendation, alerts, escalate
        """
        import os
        from pathlib import Path
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
        except ImportError:
            logger.error("LangChain not available for orchestrator reasoning")
            return {"reasoning": "LLM unavailable", "updated_recommendation": state.get("recommendation", ""),
                    "alerts": [], "escalate": False}

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
            logger.error("GOOGLE_API_KEY not set for orchestrator")
            return {"reasoning": "No API key", "updated_recommendation": state.get("recommendation", ""),
                    "alerts": [], "escalate": False}

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.4,
            max_output_tokens=2048,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )

        # Build orchestrator prompt — optimized for maximum reasoning
        # quality per token with agentic decision-making behavior
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an Orchestrator AI agent in a multi-agent clinical "
             "system for Type 2 Diabetes management.\n\n"
             "You must perform LLM-based reasoning over the full system "
             "state and produce concise, high-signal outputs.\n\n"
             "REASONING REQUIREMENT (CRITICAL):\n"
             "- You MUST perform reasoning internally before answering\n"
             "- Do NOT output full chain-of-thought\n"
             "- Output compressed reasoning conclusions only\n"
             "- Show decision justification briefly\n"
             "- You are a reasoning agent, not a formatter\n\n"
             "AGENTIC BEHAVIOR:\n"
             "- Evaluate whether current recommendations from Agent D "
             "are sufficient for this patient\n"
             "- If risk is High/Very High OR trend is worsening: "
             "Intensify recommendations OR escalate care\n"
             "- If stable/improving: Maintain plan\n"
             "- You may override or refine Agent D's recommendations "
             "based on your reasoning\n\n"
             "SAFETY RULES:\n"
             "- Do NOT prescribe medications\n"
             "- Always include: 'Consult a qualified doctor before "
             "acting on this advice'\n"
             "- Keep recommendations safe and general\n\n"
             "AVOID:\n"
             "- Long explanations or verbose medical theory\n"
             "- Listing more than 3 features\n"
             "- Repeating the explanation content\n"
             "- Copying input text verbatim\n\n"
             "OUTPUT CONSTRAINTS:\n"
             "- Max 180-220 words total\n"
             "- Use structured bullet format\n"
             "- No long paragraphs, no repetition\n"
             "- Prioritize decision-critical insights only"),
            ("human",
             "=== FULL SYSTEM STATE ===\n\n"
             "Patient Data:\n{patient_data}\n\n"
             "Risk Assessment:\n{risk_info}\n\n"
             "SHAP Explanation (Agent C):\n{explanation}\n\n"
             "Initial Recommendation (Agent D):\n{recommendation}\n\n"
             "Monitoring Trend:\n{trend_info}\n\n"
             "Active Alerts:\n{alerts_info}\n\n"
             "=== END STATE ===\n\n"
             "Produce your orchestrator output in EXACTLY this format:\n\n"
             "Risk Summary:\n"
             "- Level: <value>\n"
             "- Key Drivers: <top 3 only>\n\n"
             "Clinical Insight:\n"
             "- <1-2 lines summarizing condition>\n\n"
             "Recommendations (Refined):\n"
             "- Diet: <2 bullets>\n"
             "- Exercise: <2 bullets>\n"
             "- Lifestyle: <2 bullets>\n\n"
             "Monitoring Insight:\n"
             "- Trend: <improving / worsening / stable / first_visit>\n"
             "- Key Concern: <1-2 points>\n\n"
             "Decision:\n"
             "- Action: <Maintain / Intensify / Escalate>\n"
             "- Justification: <1-2 lines based on your reasoning>\n\n"
             "Alerts:\n"
             "- <max 2 high-impact alerts only, or 'None'>\n\n"
             "Consult a qualified doctor before acting on this advice."),
        ])

        chain = prompt | llm | StrOutputParser()

        # Format state for the prompt
        risk_info = state.get("risk", {})
        risk_str = (f"Probability: {risk_info.get('probability', 'N/A')}\n"
                    f"Risk Level: {risk_info.get('risk_level', 'N/A')}")

        trend = self.trend_analysis or {}
        trend_str = f"Overall: {trend.get('overall', 'no data')}"
        if trend.get("metric_trends"):
            for m, info in trend["metric_trends"].items():
                trend_str += f"\n  {m}: {info['direction']} ({info['pct_change']:+.1f}%)"

        alerts_str = "None"
        if self.alerts:
            alerts_str = "\n".join(f"- [{a['severity']}] {a['message']}" for a in self.alerts)

        patient_str = "\n".join(f"  {k}: {v}" for k, v in state.get("patient_data", {}).items())

        logger.info("[Orchestrator] 🧠 Using LLM for clinical reasoning …")

        try:
            result = chain.invoke({
                "patient_data": patient_str,
                "risk_info": risk_str,
                "explanation": state.get("explanation", "N/A"),
                "recommendation": state.get("recommendation", "N/A"),
                "trend_info": trend_str,
                "alerts_info": alerts_str,
            })
            logger.info("[Orchestrator] ✅ LLM reasoning complete")

            # Determine escalation from the response
            escalate = any(word in result.lower() for word in
                          ["escalat", "urgent", "immediate", "critical"])

            return {
                "reasoning": result,
                "updated_recommendation": result,
                "alerts": self.alerts,
                "escalate": escalate,
            }
        except Exception as exc:
            logger.error(f"[Orchestrator] LLM reasoning failed: {exc}")
            return {
                "reasoning": "LLM reasoning unavailable due to quota or network error. Proceeding with standard rule-based monitoring.",
                "updated_recommendation": state.get("recommendation", ""),
                "alerts": self.alerts,
                "escalate": False,
            }

    # ── Accessors ────────────────────────────────────────────────────
    def get_patient_history(self, patient_id: str = "default") -> List[Dict]:
        """Return full observation history for a patient."""
        return self.patient_history.get(patient_id, [])

    def get_all_alerts(self) -> List[Dict]:
        """Return all alerts from last generate_alerts() call."""
        return self.alerts


# ══════════════════════════════════════════════════════════════════════
# Standalone test / simulation
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    agent = MonitoringAgent()

    # ── Simulate 3 visits over time ──────────────────────────────────
    visit1 = {"Glucose": 130, "BMI": 28, "BloodPressure": 78, "Insulin": 120}
    visit2 = {"Glucose": 155, "BMI": 29, "BloodPressure": 82, "Insulin": 140}
    visit3 = {"Glucose": 190, "BMI": 31, "BloodPressure": 88, "Insulin": 180}

    print("\n" + "=" * 65)
    print("  🔁 MONITORING AGENT — Simulation")
    print("=" * 65)

    agent.update(visit1, "Moderate")
    agent.update(visit2, "High")
    agent.update(visit3, "Very High")

    # Trend
    trend = agent.analyze_trend()
    print(f"\n📊 Trend Analysis: {trend['overall']}")
    for m, info in trend.get("metric_trends", {}).items():
        print(f"   {m}: {info['direction']} ({info['pct_change']:+.1f}%)")
    print(f"   Risk trend: {trend['risk_trend']}")

    # Alerts
    alerts = agent.generate_alerts()
    print(f"\n🚨 Alerts ({len(alerts)}):")
    for a in alerts:
        print(f"   [{a['severity'].upper()}] {a['message']}")
        print(f"     → {a['action']}")

    # Adjust recommendation
    base_rec = "Diet: Eat more vegetables\nExercise: Walk 30 min daily"
    adjusted = agent.adjust_recommendation(base_rec)
    print(f"\n📋 Adjusted Recommendation:\n{adjusted}")

    print("\n" + "=" * 65)
    print("  ✅ MONITORING SIMULATION COMPLETE")
    print("=" * 65)
