"""
Personalized Recommendation Agent (Agent D) — LangChain + Google Gemini.

Generates personalized, safe, clinically relevant recommendations
based on patient data, risk level, and explainability output.

Pipeline flow:  Risk Agent → Explainability Agent → **Recommendation Agent**

Uses Google Gemini via LangChain for true agentic LLM-driven generation.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from src.config.settings import (
    RECOMMENDATIONS_BASE, RISK_CATEGORIES, MEDICAL_DISCLAIMER
)

logger = logging.getLogger(__name__)

# ── Load API key from .env ───────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── LangChain + Gemini imports ───────────────────────────────────────
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain-google-genai not installed.")

# ── Clinical Prompt Template ─────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a clinical decision-support AI assistant specializing in \
Type 2 Diabetes risk management. You provide personalized, safe, \
and actionable health recommendations.

STRICT SAFETY RULES — you MUST follow these at all times:
1. Do NOT prescribe or suggest any medications.
2. Do NOT act as a medical diagnosis — you are decision support only.
3. Do NOT suggest stopping or starting any medication.
4. Always encourage consulting a qualified healthcare provider.
5. Keep language simple, clear, and free of medical jargon.
"""

USER_PROMPT = """\
A patient has been assessed by our diabetes risk prediction model.

**Patient Data:**
{patient_data}

**Risk Level:** {risk_level}

**Explanation (top contributing factors):**
{explanation}

Based on the above, generate personalized recommendations in EXACTLY \
this format (keep each section to 3-5 bullet points):

Diet:
• <diet recommendation 1>
• <diet recommendation 2>
• <diet recommendation 3>

Exercise:
• <exercise recommendation 1>
• <exercise recommendation 2>
• <exercise recommendation 3>

Lifestyle:
• <lifestyle recommendation 1>
• <lifestyle recommendation 2>
• <lifestyle recommendation 3>

Warning:
Consult a qualified doctor before making any medical decisions. \
This system is for decision support only and is not a diagnosis.

Tailor every recommendation to this patient's specific risk level \
({risk_level}) and their key risk factors. Be specific and actionable.
"""


class RecommendationAgent:
    """
    Agent D — generates personalized health recommendations via LLM.

    Uses Google Gemini through LangChain for agentic, LLM-driven
    recommendation generation.

    Usage:
        agent = RecommendationAgent()
        result = agent.generate(patient_data, risk_level, explanation)
    """

    def __init__(self):
        """Initialize LLM and prompt chain."""
        self.llm = None
        self.chain = None
        self.llm_available = False
        self.recommendations: List[Dict] = []

        self._initialize_llm()

    # ── LLM Initialization ───────────────────────────────────────────
    def _initialize_llm(self) -> None:
        """Initialize Google Gemini LLM via LangChain."""
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain Google GenAI not installed. "
                         "Run: pip install langchain-google-genai")
            return

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
            logger.error("GOOGLE_API_KEY not set. "
                         "Add your key to .env file in project root.")
            return

        try:
            # Initialize Gemini LLM with low temperature for stable outputs
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key=api_key,
                convert_system_message_to_human=True,
            )

            # Build the LangChain prompt → LLM → parser chain
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", USER_PROMPT),
            ])
            self.chain = prompt | self.llm | StrOutputParser()

            self.llm_available = True
            logger.info("✅ Gemini LLM initialized (model: gemini-2.0-flash)")

        except Exception as exc:
            logger.error(f"Failed to initialize LLM: {exc}")
            self.llm_available = False

    # ── Core: generate() ─────────────────────────────────────────────
    def generate(self, patient_data: Dict, risk_level: str,
                 explanation: str) -> str:
        """
        Generate personalized recommendations via LLM.

        Args:
            patient_data: dict of patient features (Glucose, BMI, Age, etc.)
            risk_level:   "Low" / "Moderate" / "High" / "Very High"
            explanation:  natural language explanation from Explainability Agent

        Returns:
            Formatted recommendation string from the LLM.
        """
        if not self.llm_available:
            raise RuntimeError(
                "LLM not available. Ensure GOOGLE_API_KEY is set in .env"
            )

        # Format patient data as a readable string for the prompt
        patient_str = "\n".join(
            f"  • {k}: {v}" for k, v in patient_data.items()
        )

        logger.info(f"Generating LLM recommendations for {risk_level} risk patient...")

        try:
            result = self.chain.invoke({
                "patient_data": patient_str,
                "risk_level": risk_level,
                "explanation": explanation,
            })
            logger.info("✅ LLM recommendations generated successfully.")
        except Exception as e:
            logger.warning(f"LLM generation failed (quota/network). Falling back to template. Error: {e}")
            result = (
                f"Diet:\n• Maintain a balanced diet.\n• Monitor carbohydrate intake.\n"
                f"Exercise:\n• Engage in 30 mins of moderate exercise daily.\n"
                f"Lifestyle:\n• Regular check-ups.\n"
                f"Warning:\nConsult a qualified doctor before making any medical decisions."
            )
        return result

    # ── Pipeline interface (backward-compatible with main_pipeline) ──
    def run_pipeline(self, risk_category: str, risk_score: float,
                     sample_features: Dict, age: float,
                     explanation: str = "",
                     num_samples: int = 1) -> Dict:
        """
        Run the recommendation pipeline (compatible with main_pipeline.py).

        Args:
            risk_category: risk level string
            risk_score:    probability float
            sample_features: patient feature dict
            age:           patient age
            explanation:   explanation text from ExplainabilityAgent
            num_samples:   unused, kept for backward compat

        Returns:
            Dict with status, recommendations list, and metadata.
        """
        logger.info("=" * 50)
        logger.info("RECOMMENDATION AGENT: Starting pipeline")
        logger.info("=" * 50)

        # Map internal category keys to display names
        display_level = RISK_CATEGORIES.get(risk_category, risk_category)

        # Build explanation if not provided
        if not explanation:
            explanation = (
                f"Patient risk level is {display_level} "
                f"(probability {risk_score:.1%})."
            )

        recommendations = []

        if self.llm_available:
            # ── LLM-driven recommendation ────────────────────────
            try:
                llm_output = self.generate(
                    patient_data=sample_features,
                    risk_level=display_level,
                    explanation=explanation,
                )
                recommendations.append({
                    "risk_category": risk_category,
                    "risk_category_display": display_level,
                    "risk_score": risk_score,
                    "llm_recommendation": llm_output,
                    "disclaimer": MEDICAL_DISCLAIMER,
                    "generation_method": "LLM (Gemini)",
                })
            except Exception as exc:
                logger.error(f"LLM generation failed: {exc}")
                # Fall through to template-based
                recommendations.append(
                    self._template_recommendation(
                        risk_category, risk_score, sample_features, age
                    )
                )
        else:
            # ── Template fallback (if no API key) ────────────────
            recommendations.append(
                self._template_recommendation(
                    risk_category, risk_score, sample_features, age
                )
            )

        report = {
            "status": "success",
            "recommendations": recommendations,
            "llm_based": self.llm_available,
            "generation_method": "LLM (Gemini)" if self.llm_available else "Template",
        }

        logger.info("RECOMMENDATION AGENT: Pipeline completed")
        return report

    # ── Template fallback ─────────────────────────────────────────────
    def _template_recommendation(self, risk_category, risk_score,
                                  features, age) -> Dict:
        """Template-based fallback when LLM is unavailable."""
        base = RECOMMENDATIONS_BASE.get(risk_category,
                                         RECOMMENDATIONS_BASE.get("low", {}))
        return {
            "risk_category": risk_category,
            "risk_category_display": RISK_CATEGORIES.get(risk_category, risk_category),
            "risk_score": risk_score,
            "dietary_recommendations": base.get("diet", []),
            "exercise_recommendations": base.get("exercise", []),
            "monitoring_plan": base.get("monitoring", []),
            "disclaimer": MEDICAL_DISCLAIMER,
            "generation_method": "Template (no API key)",
        }


# ══════════════════════════════════════════════════════════════════════
# Standalone test
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    agent = RecommendationAgent()

    if not agent.llm_available:
        print("❌ LLM not available. Set GOOGLE_API_KEY in .env")
        exit(1)

    # Sample test
    patient = {
        "Glucose": 180, "BMI": 32, "Age": 50,
        "BloodPressure": 85, "Insulin": 200,
        "DiabetesPedigreeFunction": 0.6,
        "Pregnancies": 3,
    }

    print("\n" + "=" * 65)
    print("  🤖 RECOMMENDATION AGENT — Test")
    print("=" * 65)

    output = agent.generate(
        patient_data=patient,
        risk_level="High",
        explanation="High glucose and BMI are major contributing factors. "
                    "Elevated insulin and family history score also increase risk.",
    )

    print(f"\n{output}")
    print("=" * 65)
