import os
import subprocess

# 找 npx 缓存的 bing-cn-mcp 包
home = os.path.expanduser("~")
candidates = [
    os.path.join(home, "AppData", "Local", "npm-cache", "_npx"),
    os.path.join(home, "AppData", "Local", "npm", "cache", "_npx"),
    os.path.join(home, ".npm", "_npx"),
]

found = False
for c in candidates:
    if os.path.exists(c):
        print("找到 npx 缓存目录:", c)
        for root, dirs, files in os.walk(c):
            if "bing-cn" in root.lower() or "bing_cn" in root.lower():
                print("  bing-cn-mcp 包目录:", root)
                found = True
                for f in files:
                    if f.endswith((".js", ".mjs", ".cjs", ".ts")):
                        print("    文件:", f)
        if not found:
            print("  该目录下未找到 bing-cn，列顶层子目录:")
            try:
                for sub in os.listdir(c)[:20]:
                    print("    ", sub)
            except Exception as e:
                print("    读取失败", e)

# 通过 npm 查询全局安装位置
print("\n=== npm 全局 root ===")
r = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True, encoding="utf-8", timeout=10)
print(r.stdout.strip())
