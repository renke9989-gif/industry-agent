"""Run the current project consistency check from the repository root."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
python = root / "industry_research_agent" / ".venv" / "Scripts" / "python.exe"
command = [str(python if python.exists() else sys.executable), "industry_research_agent/scripts/check_agents.py"]
raise SystemExit(subprocess.call(command, cwd=root))
