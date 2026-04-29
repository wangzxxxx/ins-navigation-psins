from pathlib import Path
from dotenv import load_dotenv
PROJECT_ENV = Path('/root/.openclaw/workspace/tmp_psins_py/psins_py/.env')
if PROJECT_ENV.exists():
    load_dotenv(PROJECT_ENV, override=True)

# extracted from test_system_calibration_19pos_llm.py
SOURCE='test_system_calibration_19pos_llm.py'
METHOD='LLM assisted 19pos calibration'
