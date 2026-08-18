"""
工具模块：知识检索 + 联网搜索 + 知识自动入库
市场/竞争分析师的三个专属 Tool
"""
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from langchain_core.tools import tool
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 全局初始化（模块加载时执行一次）
# ============================================================

# 加载 embedding 模型（优先本地缓存，国内网络 huggingface 不通）
_EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
try:
    _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME, local_files_only=True)
except Exception:
    _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)

_chroma_client = chromadb.PersistentClient(
    path=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
)

_collection = _chroma_client.get_or_create_collection(
    name="industry_research_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# ============================================================
# 知识包加载路径
# 框架是行业无关的——知识包是可插拔的。
# 删除 example_* 文件夹不影响 Agent 运行（长尾行业靠联网搜索兜底）。
# 新增行业 = 在 knowledge_packs/ 下新建文件夹 + seed_knowledge.json。
# ============================================================
KNOWLEDGE_PACKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "knowledge_packs"
)


# ============================================================
# Tool 1：本地知识检索
# ============================================================

@tool
def retrieve_knowledge(query: str, industry: str = "") -> List[dict]:
    """
    从 ChromaDB 检索行业调研知识。

    Args:
        query: 检索查询，如 "宠物烘焙 市场规模 增长率"
        industry: 当前行业（用于过滤），为空则不限制

    Returns:
        检索结果列表，每项含 content / source / confidence / relevance_score
    """
    query_embedding = _embedding_model.encode([query]).tolist()

    where_filter = None
    if industry:
        where_filter = {"industry": industry}

    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return [{
            "content": "知识库暂无相关记录",
            "source": "none",
            "confidence": 0.0,
            "relevance_score": 0.0
        }]

    threshold = float(os.getenv("RETRIEVAL_THRESHOLD", "0.7"))
    output = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = 1.0 - dist
        if similarity >= threshold:
            output.append({
                "content": doc,
                "source": meta.get("source", "unknown"),
                "confidence": meta.get("confidence", 0.5),
                "relevance_score": round(similarity, 4),
                "industry": meta.get("industry", ""),
                "topic": meta.get("topic", ""),
                "key_data": meta.get("key_data", []),
            })
        else:
            output.append({
                "content": f"检索结果相似度 {similarity:.2f} 低于阈值 {threshold}，建议联网搜索",
                "source": "low_confidence",
                "confidence": 0.0,
                "relevance_score": round(similarity, 4)
            })

    for meta in metadatas:
        if meta:
            meta["access_count"] = meta.get("access_count", 0) + 1
            meta["last_accessed"] = datetime.now().isoformat()

    return output if output else [{
        "content": "知识库暂无该行业的调研数据",
        "source": "none",
        "confidence": 0.0,
        "relevance_score": 0.0
    }]


# ============================================================
# Tool 2：联网搜索（本地未命中时自动触发）
# ============================================================

@tool
def web_search(query: str, industry: str = "") -> List[dict]:
    """
    联网搜索行业调研数据（真实网页搜索，免费、无需 Key）。
    当本地 RAG 未命中时由 Agent 自动调用。

    Args:
        query: 搜索关键词
        industry: 当前行业（自动拼接到搜索词中）

    Returns:
        搜索结果列表，每项含 title / body / href
    """
    try:
        from ddgs import DDGS
    except ImportError:
        # ddgs 未安装时降级到旧版 duckduckgo_search
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return [{"title": "搜索不可用", "body": "缺少 ddgs 依赖，请执行: pip install ddgs", "href": ""}]

    full_query = f"{industry} {query}" if industry else query
    # 附加上下文词，让搜索结果更贴合行业调研场景
    search_query = f"{full_query} 市场规模 增长率 报告"

    # 代理设置（国内网络访问外网可能需要）
    proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")

    try:
        results = []
        # 固定用 html 后端（走 html.duckduckgo.com，国内网络已验证可通）
        # 后端优先级：html(最稳) → lite(轻量) → duckduckgo
        backends = ["html", "lite", "duckduckgo"]
        last_error = None

        for backend in backends:
            try:
                kwargs = {
                    "region": "cn-zh",
                    "safesearch": "off",
                    "max_results": 5,
                    "backend": backend,
                }
                if proxy:
                    kwargs["proxy"] = proxy

                with DDGS() as ddgs:
                    for r in ddgs.text(search_query, **kwargs):
                        results.append({
                            "title": r.get("title", ""),
                            "body": r.get("body", ""),
                            "href": r.get("href", ""),
                        })
                if results:
                    break
            except Exception as e:
                last_error = e
                continue

        if not results:
            err_msg = f"联网搜索「{search_query}」未返回结果"
            if last_error:
                err_msg += f"（{str(last_error)[:80]}）"
            return [{"title": "无结果", "body": err_msg, "href": ""}]

        return results

    except Exception as e:
        return [{"title": "搜索异常", "body": f"联网搜索失败: {str(e)}", "href": ""}]


# ============================================================
# Tool 3：知识自动入库（搜索后触发）
# ============================================================

@tool
def knowledge_upsert(
    industry: str,
    topic: str,
    content: str,
    key_data: Optional[List[dict]] = None,
    source: str = "web_search",
    source_url: str = "",
    region: str = "全国",
    related_dimensions: Optional[List[str]] = None
) -> dict:
    """
    将结构化行业调研知识自动写入 ChromaDB。联网搜索后由 Agent 调用。

    Args:
        industry: 行业名称，如 "宠物烘焙"
        topic: 主题，如 "市场规模""竞争格局""商业模式"
        content: 调研内容描述
        key_data: 关键数据点 [{"metric":"年增速","value":"15%","year":"2026"}]
        source: 来源类型
        source_url: 原始链接
        region: 适用区域
        related_dimensions: 关联调研维度

    Returns:
        {"success": bool, "action": "inserted"|"updated"|"duplicate", "id": str}
    """
    text = f"行业:{industry}\n主题:{topic}\n内容:{content}\n数据:{json.dumps(key_data or [], ensure_ascii=False)}"

    new_embedding = _embedding_model.encode([text]).tolist()
    existing = _collection.query(
        query_embeddings=new_embedding,
        n_results=1
    )

    if existing["distances"] and existing["distances"][0]:
        similarity = 1.0 - existing["distances"][0][0]
        if similarity > 0.85 and existing["ids"][0]:
            existing_id = existing["ids"][0][0]
            _collection.update(
                ids=[existing_id],
                metadatas=[{
                    "access_count": existing["metadatas"][0][0].get("access_count", 0) + 1,
                    "last_accessed": datetime.now().isoformat()
                }]
            )
            return {"success": True, "action": "updated", "id": existing_id}

    knowledge_id = str(uuid.uuid4())[:8]
    _collection.add(
        ids=[knowledge_id],
        documents=[text],
        embeddings=new_embedding,
        metadatas=[{
            "industry": industry,
            "topic": topic,
            "key_data": json.dumps(key_data or [], ensure_ascii=False),
            "region": region,
            "source": source,
            "source_url": source_url,
            "confidence": 0.9 if source == "manual" else 0.5,
            "access_count": 1,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "related_dimensions": json.dumps(related_dimensions or [], ensure_ascii=False)
        }]
    )

    return {"success": True, "action": "inserted", "id": knowledge_id}


# ============================================================
# 知识库初始化：加载种子知识
# ============================================================

def load_seed_knowledge():
    """
    启动时加载所有知识包的种子知识到 ChromaDB。

    加载规则：
    - 跳过 _template/（接口规范模板，不含实际数据）
    - 其余全部加载（含 example_* 示例知识包）
    - 知识包是可插拔的——删除任意文件夹不影响 Agent 运行，长尾行业靠联网搜索兜底
    - 新增行业 = 在 knowledge_packs/ 下新建文件夹 + seed_knowledge.json
    """
    total_loaded = 0
    for pack_dir in os.listdir(KNOWLEDGE_PACKS_DIR):
        if pack_dir.startswith("_"):
            continue
        pack_path = os.path.join(KNOWLEDGE_PACKS_DIR, pack_dir)
        seed_file = os.path.join(pack_path, "seed_knowledge.json")
        if os.path.isfile(seed_file):
            with open(seed_file, "r", encoding="utf-8") as f:
                seeds = json.load(f)
            for item in seeds:
                knowledge_upsert.invoke({
                    "industry": item.get("industry", pack_dir),
                    "topic": item.get("topic", ""),
                    "content": item.get("content", ""),
                    "key_data": item.get("key_data", []),
                    "source": "manual",
                    "region": item.get("region", "全国"),
                    "related_dimensions": item.get("related_dimensions", [])
                })
                total_loaded += 1
    print(f"[KnowledgeBase] 已加载 {total_loaded} 条种子知识（可插拔，删除不影响运行）")
    return total_loaded
