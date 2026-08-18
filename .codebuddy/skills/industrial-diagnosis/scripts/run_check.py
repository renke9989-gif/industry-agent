"""一键环境检查"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "industrial_diagnosis_agent"))
from scripts.agent_tools import check_environment
import json
print(json.dumps(check_environment(), indent=2, ensure_ascii=False))
