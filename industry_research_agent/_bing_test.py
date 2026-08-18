import urllib.request
import time

urls = {
    "cn.bing.com 首页": "https://cn.bing.com",
    "cn.bing.com 搜索": "https://cn.bing.com/search?q=宠物烘焙",
    "www.bing.com 搜索": "https://www.bing.com/search?q=test",
}

print("=== 测试必应访问 ===")
for name, url in urls.items():
    try:
        start = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        elapsed = time.time() - start
        print(f"[OK] {name}: 状态 {resp.status}, 耗时 {elapsed:.1f}s")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:80]}")
