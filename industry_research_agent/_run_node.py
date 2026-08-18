import os
import subprocess

d = "D:/"
t = [x for x in os.listdir(d) if x.startswith("AI")][0]
base = os.path.join(d, t, "industry-agent", "industry_research_agent")
os.chdir(base)

# 在正确的工作目录下运行 node 脚本（相对路径避免中文路径问题）
result = subprocess.run(
    ["node", "_node_test.js"],
    cwd=base,
    capture_output=True,
    text=True,
    encoding="utf-8",
    timeout=40,
)
print(result.stdout)
print(result.stderr)
