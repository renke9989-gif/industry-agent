import os
import subprocess

# 定位 bing-cn-mcp 包
home = os.path.expanduser("~")
npx_cache = os.path.join(home, "AppData", "Local", "npm-cache", "_npx")
pkg_dir = None
for root, dirs, files in os.walk(npx_cache):
    if "bing-cn-mcp" in root and root.endswith("bing-cn-mcp"):
        pkg_dir = root
        break

if not pkg_dir:
    print("未找到 bing-cn-mcp 包")
else:
    print("包目录:", pkg_dir)
    build_dir = os.path.join(pkg_dir, "build")
    print("build 目录文件:", os.listdir(build_dir))

    # 写一个测试脚本到包目录（能 require 到 axios）
    test_js = os.path.join(pkg_dir, "_test_axios.js")
    js_code = '''
const axios = require('axios');
async function test() {
    try {
        const r = await axios.get('https://cn.bing.com/search', {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'zh-CN,zh;q=0.9',
            },
            params: { q: '宠物烘焙 市场规模' },
            timeout: 15000,
        });
        console.log('RESULT: OK status=' + r.status + ' len=' + r.data.length);
    } catch (e) {
        console.log('RESULT: FAIL code=' + (e.code || '') + ' msg=' + (e.message || '').slice(0,120));
    }
}
test();
'''
    with open(test_js, "w", encoding="utf-8") as f:
        f.write(js_code)

    r = subprocess.run(
        ["node", "_test_axios.js"],
        cwd=pkg_dir,
        capture_output=True, text=True, encoding="utf-8", timeout=40,
    )
    print("stdout:", r.stdout)
    print("stderr:", r.stderr[:500])

    # 清理
    os.remove(test_js)
