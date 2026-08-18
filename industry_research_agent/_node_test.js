// 测试 Node.js 原生 fetch 访问必应
async function test() {
    console.log('=== 测试 Node fetch 访问必应 ===');
    
    // 1. 直连 www.bing.com
    try {
        const r = await fetch('https://www.bing.com/search?q=test', { signal: AbortSignal.timeout(10000) });
        const text = await r.text();
        console.log('[OK] www.bing.com 直连:', r.status, '长度:', text.length);
    } catch (e) {
        console.log('[FAIL] www.bing.com 直连:', e.cause ? e.cause.code : e.message);
    }
    
    // 2. 直连 cn.bing.com
    try {
        const r = await fetch('https://cn.bing.com/search?q=test', { signal: AbortSignal.timeout(10000) });
        const text = await r.text();
        console.log('[OK] cn.bing.com 直连:', r.status, '长度:', text.length);
    } catch (e) {
        console.log('[FAIL] cn.bing.com 直连:', e.cause ? e.cause.code : e.message);
    }
}

test();
