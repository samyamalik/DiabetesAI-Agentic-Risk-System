# 🏥 Diabetes AI Risk Stratification System

## Multi-Agent Agentic AI for Type 2 Diabetes Risk Stratification & Personalized Management

A production-ready Python system implementing a **5-agent agentic architecture** for diabetes risk prediction, explainable AI insights, personalized recommendations, and longitudinal patient monitoring.

### 🎯 Overview

This system demonstrates a sophisticated multi-agent collaboration approach where:
- **Ingestion Agent** prepares and validates patient data
- **Risk Agent** predicts diabetes risk using ML models
- **Explainability Agent** provides SHAP-based interpretability
- **Recommendation Agent** generates personalized care plans
- **Monitoring Agent** tracks patients over time and triggers alerts

All agents work together in an orchestrated pipeline to provide comprehensive diabetes risk assessment and management support.

### ✨ Key Features

✅ **Multi-Agent Architecture** - 5 specialized agents collaborating autonomously  
✅ **Risk Stratification** - Low/Moderate/High/Very High risk classification  
✅ **Explainable AI** - SHAP-based feature importance & natural language explanations  
✅ **Personalized Recommendations** - Diet, exercise, and monitoring plans  
✅ **Longitudinal Monitoring** - Track patient trends and generate alerts  
✅ **Production-Ready** - Clean, modular, well-documented code  
✅ **Interactive UI** - Streamlit frontend for easy interaction  
✅ **Ethical Safeguards** - Medical disclaimers and decision-support warnings  

---

## 📁 Project Structure

```
AgentMinor/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── ingestion_agent.py       # Data loading & preprocessing
│   │   ├── risk_agent.py             # Risk prediction
│   │   ├── explainability_agent.py   # SHAP explanations
│   │   ├── recommendation_agent.py   # Personalized recommendations
│   │   └── monitoring_agent.py       # Longitudinal tracking
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                  # Model training
│   │   ├── predict.py                # Model inference
│   │   └── saved_models/             # Persisted models
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocessing.py          # Data preprocessing utilities
│   │   └── feature_engineering.py    # Feature engineering utilities
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py               # Centralized configuration
│   ├── __init__.py
│   └── main_pipeline.py              # Main orchestration pipeline
├── app/
│   └── streamlit_app.py              # Web interface
├── data/
│   ├── raw/                          # Raw datasets
│   └── processed/                    # Processed datasets
├── vector_db/                        # Vector database for RAG
├── notebooks/
│   ├── eda.ipynb                     # Exploratory Data Analysis
│   └── model_training.ipynb          # Model Training Demo
├── logs/
│   └── diabetes_ai_system.log        # System logs
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

---

## 🤖 Agent Architecture

### 1. Ingestion Agent (`src/agents/ingestion_agent.py`)

**Responsibilities:**
- Load diabetes datasets (CSV format)
- Handle missing values (mean, median, forward-fill strategies)
- Detect outliers (IQR and Z-score methods)
- Perform feature engineering
- Validate data quality

**Key Methods:**
```python
agent = IngestionAgent()
agent.load_dataset(file_path)
agent.handle_missing_values(strategy="mean")
agent.detect_outliers(method="iqr")
agent.perform_feature_engineering()
results = agent.run_pipeline()
```

### 2. Risk Agent (`src/agents/risk_agent.py`)

**Responsibilities:**
- Load trained ML models (Random Forest, XGBoost, Logistic Regression)
- Generate risk probability scores
- Classify patients into 4 risk categories
- Output confidence metrics

**Key Methods:**
```python
agent = RiskAgent()
agent.load_model(model_path)
risk_scores = agent.predict_risk(X)
categories = agent.classify_risk(risk_scores)
report = agent.run_pipeline(X)
```

### 3. Explainability Agent (`src/agents/explainability_agent.py`)

**Responsibilities:**
- Generate SHAP values for model interpretability
- Calculate feature importance scores
- Create natural language explanations
- Provide clinical context for decisions

**Key Methods:**
```python
agent = ExplainabilityAgent()
agent.initialize_shap(model, X_background)
shap_report = agent.generate_shap_values(model, X)
explanation = agent.generate_natural_language_explanation(...)
results = agent.run_pipeline(model, X, risk_scores, categories)
```

### 4. Recommendation Agent (`src/agents/recommendation_agent.py`)

**Responsibilities:**
- Generate personalized diet recommendations
- Create customized exercise plans
- Provide risk-specific clinical advice
- Include medical disclaimers and safety warnings

**Key Methods:**
```python
agent = RecommendationAgent()
diet = agent.personalize_diet_recommendation(risk_category, features)
exercise = agent.personalize_exercise_recommendation(risk_category, features, age)
advice = agent.generate_risk_specific_advice(risk_category, features)
recommendations = agent.run_pipeline(risk_category, risk_score, features, age)
```

### 5. Monitoring Agent (`src/agents/monitoring_agent.py`)

**Responsibilities:**
- Track patient metrics over time
- Detect metric deterioration
- Generate automated alerts
- Adapt recommendations based on trends

**Key Methods:**
```python
agent = MonitoringAgent()
agent.add_patient_measurement(patient_id, date, metrics)
trends = agent.analyze_trends(patient_id, metric_name)
alerts = agent.detect_alerts(patient_id, current_risk, previous_risk)
results = agent.run_pipeline(patient_id, current_risk, metrics)
```

---

## 🔁 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Agent Orchestration Pipeline                 │
└─────────────────────────────────────────────────────────────────┘

INPUT: Patient Data (CSV or Form Input)
   │
   ▼
┌─────────────────────────────────────────┐
│     1. INGESTION AGENT                  │
│  ├─ Load dataset                        │
│  ├─ Handle missing values               │
│  ├─ Detect outliers                     │
│  ├─ Feature engineering                 │
│  └─ Data quality validation             │
└─────────────────────────────────────────┘
   │
   ├─ OUTPUT: Preprocessed Data
   │
   ▼
┌─────────────────────────────────────────┐
│     2. RISK AGENT                       │
│  ├─ Load ML model                       │
│  ├─ Generate predictions                │
│  ├─ Calculate probability scores        │
│  └─ Classify risk levels                │
└─────────────────────────────────────────┘
   │
   ├─ OUTPUT: Risk Scores & Categories
   │
   ▼
┌─────────────────────────────────────────┐
│     3. EXPLAINABILITY AGENT             │
│  ├─ Generate SHAP values                │
│  ├─ Calculate feature importance        │
│  ├─ Create natural language              │
│  │  explanations                        │
│  └─ Provide clinical context            │
└─────────────────────────────────────────┘
   │
   ├─ OUTPUT: Explanations & Feature Importance
   │
   ▼
┌─────────────────────────────────────────┐
│     4. RECOMMENDATION AGENT             │
│  ├─ Diet recommendations                │
│  ├─ Exercise plans                      │
│  ├─ Risk-specific advice                │
│  └─ Medical disclaimers                 │
└─────────────────────────────────────────┘
   │
   ├─ OUTPUT: Personalized Recommendations
   │
   ▼
┌─────────────────────────────────────────┐
│     5. MONITORING AGENT                 │
│  ├─ Track patient history               │
│  ├─ Detect deterioration                │
│  ├─ Generate alerts                     │
│  └─ Adapt recommendations               │
└─────────────────────────────────────────┘
   │
   ▼
OUTPUT: Comprehensive Risk Assessment Report
  ├─ Risk Level & Probability
  ├─ Feature Importance Explanation
  ├─ Personalized Care Plan
  ├─ Monitoring Recommendations
  └─ Clinical Alerts
```

---

## 🚀 Quick Start

### Installation

1. **Clone or download the project:**
```bash
cd AgentMinor
```

2. **Create a Python virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Pipeline

#### Option 1: Run Complete Pipeline (Recommended)

```bash
python -c "from src.main_pipeline import DiabetesAIAgentsPipeline; pipeline = DiabetesAIAgentsPipeline(); results = pipeline.run_complete_pipeline(train_model=True); print('Pipeline complete!')"
```

#### Option 2: Run Individual Components

```python
# Import pipeline
from src.main_pipeline import DiabetesAIAgentsPipeline

# Create pipeline
pipeline = DiabetesAIAgentsPipeline()

# Run stages individually
ingestion_results, processed_data = pipeline.stage_1_data_ingestion()
risk_results, risk_scores, risk_categories, X = pipeline.stage_2_risk_prediction(processed_data)
explainability_results = pipeline.stage_3_explainability(X, risk_scores, risk_categories)
recommendations = pipeline.stage_4_personalized_recommendations(risk_scores, risk_categories, X)
monitoring = pipeline.stage_5_monitoring_and_alerts(risk_scores, risk_categories, X)
```

#### Option 3: Launch Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

#### Option 4: Run Notebooks

```bash
jupyter notebook notebooks/eda.ipynb
jupyter notebook notebooks/model_training.ipynb
```

---

## 📊 Sample Output

### Risk Assessment Result

```
================================================================================
RISK ASSESSMENT RESULT
================================================================================

📊 RISK LEVEL: High Risk
💯 PROBABILITY: 0.82 (82%)

🔍 KEY FACTORS (SHAP-based):
  1. Fasting Glucose: 145 mg/dL (High importance)
  2. BMI: 32.5 (Obesity range)
  3. Age: 58 years
  4. Blood Pressure: 145/92 mmHg (Elevated)

💡 RECOMMENDATIONS:
  🍽️ Diet:
    • Reduce sugar and refined carbs
    • Increase fiber intake
    • Follow Mediterranean or DASH diet
    • Consult with a registered dietitian

  🏃 Exercise:
    • 200 minutes of moderate-intensity activity per week
    • Include resistance training
    • Start with medical clearance

  ⚕️ Clinical Advice:
    • Schedule appointment with primary care physician
    • Consider referral to endocrinologist
    • Intensive lifestyle modification program recommended
    • Regular glucose and metabolic monitoring essential

📅 MONITORING PLAN:
    • Checkup frequency: Every 3 months
    • Glucose testing: Every 3-6 months
    • Weight monitoring: Monthly
    • Lifestyle review: Monthly

⚠️ ALERTS:
    🚨 HIGH RISK - Urgent intervention needed
    • Risk score has increased by 8% over last month
    • Glucose levels trending upward

================================================================================
```

---

## 🔐 Ethical Considerations

This system includes critical safeguards:

⚠️ **Medical Disclaimer:**
```
This AI system is intended for DECISION SUPPORT ONLY and is NOT a medical diagnosis tool.
- Predictions are based on statistical models and should not replace clinical judgment.
- Always consult with qualified healthcare providers before taking medical actions.
- This system is not a substitute for professional medical advice, diagnosis, or treatment.
- For medical emergencies, contact your local emergency services immediately.
```

✅ **Key Ethical Constraints:**
- System explicitly identifies itself as decision-support, not diagnosis
- All recommendations include medical disclaimers
- Safe handling of missing/low-quality data
- Transparent feature importance through SHAP
- Bias detection and mitigation strategies
- Privacy-preserving data handling

---

## 🧪 Testing the System

### Run Tests

```bash
pytest tests/
```

### Unit Tests

```bash
python -m pytest tests/test_agents.py -v
python -m pytest tests/test_models.py -v
python -m pytest tests/test_pipeline.py -v
```

### Manual Testing

```python
# Test individual agents
from src.agents import IngestionAgent, RiskAgent

# Create test data
agent = IngestionAgent()
data = agent.load_dataset()

# Test preprocessing
agent.handle_missing_values()
agent.detect_outliers()

# Risk prediction
risk_agent = RiskAgent()
risk_agent.load_model()
predictions = risk_agent.predict_risk(data)
```

---

## 📊 Model Training

To train a new model:

```bash
python -c "from src.models.train import ModelTrainer; trainer = ModelTrainer(); results = trainer.run_training_pipeline(); print(results)"
```

Or use the Jupyter notebook:

```bash
jupyter notebook notebooks/model_training.ipynb
```

### Supported Models

1. **Logistic Regression** - Fast, interpretable baseline
2. **Random Forest** - Ensemble with feature importance
3. **XGBoost** - Gradient boosting with high performance

The pipeline automatically selects the best performing model.

---

## 🛠️ Configuration

All settings are centralized in `src/config/settings.py`:

```python
# Paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"

# Model parameters
MODEL_TYPE = "xgboost"
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1
}

# Risk thresholds
RISK_THRESHOLDS = {
    "low": (0.0, 0.25),
    "moderate": (0.25, 0.50),
    "high": (0.50, 0.75),
    "very_high": (0.75, 1.0)
}

# Feature engineering
FEATURE_COLUMNS = [
    "age", "gender", "bmi", "blood_pressure_systolic",
    "blood_pressure_diastolic", "fasting_glucose", "cholesterol",
    "triglycerides", "physical_activity", "smoking_status",
    "alcohol_consumption", "family_history"
]
```

---

## 📈 Performance Metrics

The system tracks:
- Model accuracy & AUC-ROC
- Prediction latency
- Data quality metrics
- Alert generation rate
- Recommendation coverage

---

## 🔗 Integration Points

### LLM Integration (Optional)

To enable LLM-based recommendations:

1. Set up OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. The system will automatically use LLM if available

### Vector Database

To enable RAG capabilities:

```python
# Install additional dependencies
pip install chroma-db  # OR faiss-gpu

# Vector DB is initialized in recommendation_agent
```

---

## 📝 Logging

All activities are logged to `logs/diabetes_ai_system.log`:

```
2024-01-15 10:30:45,123 - src.agents.ingestion_agent - INFO - Loading dataset...
2024-01-15 10:30:47,456 - src.agents.risk_agent - INFO - Generating predictions...
2024-01-15 10:30:52,789 - src.agents.explainability_agent - INFO - Computing SHAP values...
```

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: SHAP not installing**
```bash
pip install --upgrade shap
# If still fails, try conda
conda install -c conda-forge shap
```

**Issue: XGBoost import error**
```bash
pip install --upgrade xgboost
```

**Issue: Streamlit not running**
```bash
pip install --upgrade streamlit
streamlit run app/streamlit_app.py --logger.level=debug
```

**Issue: Model not found**
- Train a new model: `python -c "from src.models.train import ModelTrainer; ModelTrainer().run_training_pipeline()"`
- Or place pre-trained model in `src/models/saved_models/`

---

## 📚 Documentation

### Architecture Diagram

See `docs/architecture.md` for detailed architecture documentation.

### API Reference

Comprehensive API documentation for all agents:
- [Ingestion Agent API](docs/api/ingestion_agent.md)
- [Risk Agent API](docs/api/risk_agent.md)
- [Explainability Agent API](docs/api/explainability_agent.md)
- [Recommendation Agent API](docs/api/recommendation_agent.md)
- [Monitoring Agent API](docs/api/monitoring_agent.md)

### Use Cases

- [Patient Risk Assessment](docs/use_cases/risk_assessment.md)
- [Longitudinal Monitoring](docs/use_cases/monitoring.md)
- [Batch Prediction](docs/use_cases/batch_prediction.md)

---

## 📦 Dependencies

### Core Dependencies
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (ML models)
- xgboost (gradient boosting)

### Explainability
- shap (feature importance)

### Recommendations
- langchain (LLM integration)
- faiss-cpu (vector similarity)

### Frontend
- streamlit (web interface)

### Development
- jupyter (notebooks)
- pytest (testing)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and commit: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙋 Support & Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## 📚 References

### Research Papers
- [SHAP: A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [Diabetes Risk Factors and Prediction Models](https://pubmed.ncbi.nlm.nih.gov/)

### Clinical Guidelines
- ADA Standards of Care in Diabetes
- CDC Diabetes Prevention Program
- WHO Diabetes Guidelines

---

## ✅ Checklist for Deployment

- [ ] All agents tested individually
- [ ] End-to-end pipeline validated
- [ ] Models trained and saved
- [ ] Logging configured
- [ ] Medical disclaimers in place
- [ ] Security review completed
- [ ] Documentation updated
- [ ] Dependencies locked in requirements.txt
- [ ] Unit tests passing
- [ ] Performance benchmarks acceptable

---

## 🎓 Educational Value

This project is designed to teach:
1. **Multi-Agent Architecture** - Design patterns for agent collaboration
2. **Machine Learning Ops** - Model training, deployment, monitoring
3. **Explainable AI** - SHAP and feature importance
4. **Healthcare AI** - Responsible AI in medical domain
5. **Software Engineering** - Clean code, modularity, testing

---

**Last Updated:** April 2024  
**Version:** 1.0.0  
**Status:** Production Ready ✅
