import json, statistics
from pathlib import Path

games = json.loads(Path("data/processed/games.json").read_text(encoding="utf-8"))
lens = [len(g.split()) for g in games]
print("партий:", len(games))
print("пример строки:", games[0][:100])
print("средняя длина:", statistics.mean(lens), "| медиана:", statistics.median(lens), "| макс:", max(lens))
print("примеров будет примерно:", sum(l - 12 for l in lens))