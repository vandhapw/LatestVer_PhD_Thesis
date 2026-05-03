"""Quick inventory of Ver13/*.py — categorize by deps + main."""
import re
from pathlib import Path

ROOT = Path(r"D:\AI-LLM\Claude Experiment\Ver13")

def analyze(p: Path):
    text = p.read_text(encoding="utf-8", errors="replace")
    return {
        "name": p.name,
        "loc": len(text.splitlines()),
        "has_main": bool(re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", text)),
        "needs_mongo": bool(re.search(r"pymongo|MongoClient", text)),
        "needs_kafka": bool(re.search(r"\bkafka\b|KafkaConsumer|KafkaProducer", text)),
        "needs_ollama": bool(re.search(r"ollama|Ollama", text)),
        "needs_torch": bool(re.search(r"\bimport\s+torch\b|from\s+torch", text)),
        "needs_qiskit": bool(re.search(r"qiskit", text)),
        "first_docstring": (re.search(r'"""(.+?)"""', text, re.S).group(1).strip().split("\n")[0][:80] if re.search(r'"""(.+?)"""', text, re.S) else "no docstring"),
    }

files = sorted(ROOT.glob("*.py"))
files = [f for f in files if f.name != "_inventory.py"]
results = [analyze(p) for p in files]

print(f"Total: {len(results)} files\n")
print(f"{'name':40s} {'LOC':>5s} {'main':>5s} {'mongo':>6s} {'kafka':>6s} {'ollama':>7s} {'torch':>6s} {'qiskit':>7s}  docstring")
print("-" * 145)
for r in results:
    print(f"{r['name']:40s} {r['loc']:5d} {str(r['has_main']):>5s} {str(r['needs_mongo']):>6s} {str(r['needs_kafka']):>6s} {str(r['needs_ollama']):>7s} {str(r['needs_torch']):>6s} {str(r['needs_qiskit']):>7s}  {r['first_docstring']}")

print("\n=== Local-safe (entry, no external deps) ===")
local_safe = [r for r in results if r["has_main"] and not (r["needs_mongo"] or r["needs_kafka"] or r["needs_ollama"])]
for r in local_safe:
    print(f"  - {r['name']}: {r['first_docstring']}")

print(f"\n=== Need external services ({sum(1 for r in results if r['has_main'] and (r['needs_mongo'] or r['needs_kafka'] or r['needs_ollama']))}) ===")
for r in results:
    if r["has_main"] and (r["needs_mongo"] or r["needs_kafka"] or r["needs_ollama"]):
        deps = []
        if r["needs_mongo"]: deps.append("mongo")
        if r["needs_kafka"]: deps.append("kafka")
        if r["needs_ollama"]: deps.append("ollama")
        print(f"  - {r['name']}: needs {','.join(deps)}")
