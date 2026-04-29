from pathlib import Path
from dotenv import load_dotenv
PROJECT_ENV = Path('/root/.openclaw/workspace/tmp_psins_py/psins_py/.env')
if PROJECT_ENV.exists():
    load_dotenv(PROJECT_ENV, override=True)

# extracted from test_calibration_llm_hyperparam_tuner.py
SOURCE='test_calibration_llm_hyperparam_tuner.py'
METHOD='LLM hyperparameter tuner'
