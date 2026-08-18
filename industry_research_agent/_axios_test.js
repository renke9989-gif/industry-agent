// 用 axios 复现必应 MCP 的请求，看是否超时
const axios = require('axios');

async function test() {
    console.log('=== 用 axios 复现必应 MCP 的请求 ===');
    const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';
    
    try {
        const response = await axios.get('https://cn.bing.com/search', {
            headers: {
                'User-Agent': USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            params: { q: '宠物烘焙 市场规模' },
            timeout: 15000,
        });
        console.log('[OK] axios 访问 cn.bing.com/search:', response.status, '长度:', response.data.length);
    } catch (e) {
        console.log('[FAIL] axios 访问失败:', e.code || e.message);
        if (e.response) console.log('  状态码:', e.response.status);
    }
}

test();
