from pathlib import Path
from dotenv import load_dotenv
PROJECT_ENV = Path('/root/.openclaw/workspace/tmp_psins_py/psins_py/.env')
if PROJECT_ENV.exists():
    load_dotenv(PROJECT_ENV, override=True)

# extracted from calibration_path_optimizer_llm.py
SOURCE='calibration_path_optimizer_llm.py'
METHOD='LLM calibration path optimizer'
