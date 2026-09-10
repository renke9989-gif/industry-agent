from pathlib import Path
import sys
from tempfile import TemporaryDirectory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from private_rag import PrivateDocumentStore

def run() -> dict:
    with TemporaryDirectory() as tmp:
        root=Path(tmp); (root/'process.md').write_text('LPBF 激光功率与扫描速度需要在设备边界内验证。氩气保护可降低氧含量风险。', encoding='utf-8')
        store=PrivateDocumentStore(root/'rag.sqlite3', chunk_chars=300)
        indexed=store.index_dir(root); hits=store.search('氩气保护氧含量', top_k=2)
        passed=bool(indexed and hits and hits[0]['path'].endswith('process.md'))
        return {'passed':passed,'documents':len(indexed),'hits':len(hits)}

if __name__ == '__main__': print(run())
