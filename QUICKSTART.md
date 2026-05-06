# Quick Start Guide

## Setup (1 minute)

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the main pipeline
python -c "from src.main_pipeline import DiabetesAIAgentsPipeline; pipeline = DiabetesAIAgentsPipeline(); results = pipeline.run_complete_pipeline(train_model=True)"
```

## Usage Options

### Option 1: Python Pipeline
```bash
python -m src.main_pipeline
```

### Option 2: Jupyter Notebooks
```bash
jupyter notebook notebooks/eda.ipynb
jupyter notebook notebooks/model_training.ipynb
```

### Option 3: Streamlit Web App
```bash
streamlit run app/streamlit_app.py
# Open http://localhost:8501 in your browser
```

### Option 4: Python Script
```python
from src.main_pipeline import DiabetesAIAgentsPipeline

# Create pipeline
pipeline = DiabetesAIAgentsPipeline()

# Run complete pipeline
results = pipeline.run_complete_pipeline()
```

## Troubleshooting

**SHAP installation fails:**
```bash
pip install --upgrade shap
# Or use conda: conda install -c conda-forge shap
```

**XGBoost import error:**
```bash
pip install --upgrade xgboost
```

**Model not found:**
The first run will automatically train a model. Subsequent runs will use the saved model.

## File Structure

```
src/
  agents/              # 5 agent modules
  models/              # ML training/prediction
  utils/               # Preprocessing & feature engineering
  config/              # Configuration settings
  main_pipeline.py     # Orchestration
app/                   # Streamlit frontend
data/                  # Datasets
notebooks/             # Jupyter notebooks
logs/                  # System logs
```

## Next Steps

1. ✅ Run the pipeline: `python -m src.main_pipeline`
2. ✅ Explore data: `jupyter notebook notebooks/eda.ipynb`
3. ✅ Train models: `jupyter notebook notebooks/model_training.ipynb`
4. ✅ Use web UI: `streamlit run app/streamlit_app.py`

## Documentation

See [README.md](README.md) for comprehensive documentation.
