from datasets import load_dataset
import json
from pathlib import Path

# streaming=True: частичная скачка
ds = load_dataset("angeluriot/chess_games", split="train", streaming=True)

MIN_ELO = 2100            # мин эло
MIN_PLIES = 24            # мин. 24 полухода (12 ходов) в партии
TARGET = 200_000          # столько партий хотим собрать

games = []
for game in ds:
    try:
        w = int(game["white_elo"])
        b = int(game["black_elo"])
    except (KeyError, TypeError, ValueError):
        continue
    if w < MIN_ELO or b < MIN_ELO:
        continue
    moves = game.get("moves_uci") or [] # чтобы не упало на none
    if len(moves) < MIN_PLIES:              #отсев коротких партий
        continue
    games.append(" ".join(moves))
    if len(games) >= TARGET:
        break

print(f"партий: {len(games)}")
OUT = Path("data/processed/games.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(games, ensure_ascii=False), encoding="utf-8")
print(f"сохранено: {OUT}, размер {OUT.stat().st_size / 1e6:.1f} МБ")