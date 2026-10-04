import pandas as pd
import numpy as np
from pathlib import Path
import json

META = Path("data/raw/output_game_info.csv")
MOVES = Path("data/raw/output_moves.csv")

# Читаем метаданные
meta = pd.read_csv(
    META,
    usecols=["game_id", "time_control", "white_elo", "black_elo", "termination", "result"],
)

# Парсим time_control в секунды
def parse_time_control(tc):
    """'300+5' -> 300, '60+0' -> 60, '1800+30' -> 1800"""
    try:
        return int(str(tc).split("+")[0])
    except:
        return None

meta["time_seconds"] = meta["time_control"].apply(parse_time_control)

# Конвертируем elo в числа (невалидные станут NaN)
meta["white_elo"] = pd.to_numeric(meta["white_elo"], errors="coerce")
meta["black_elo"] = pd.to_numeric(meta["black_elo"], errors="coerce")

# Определяем режим
def classify_time(seconds):
    if seconds is None or pd.isna(seconds):
        return "unknown"
    if seconds < 180:
        return "bullet"
    elif seconds < 900:
        return "blitz"
    elif seconds < 1500:
        return "rapid"
    else:
        return "classical"

meta["mode"] = meta["time_seconds"].apply(classify_time)

# Смотрим распределение режимов
print("=== Распределение режимов ===")
print(meta["mode"].value_counts())

# Фильтр: classical + оба ЭЛО >= 1800 (понижаем порог, чтобы было больше партий)
mask = (
    (meta["mode"] == "classical")
    & (meta["white_elo"] >= 1800)
    & (meta["black_elo"] >= 1800)
    & (meta["termination"].isin(["Normal"]))  # опционально: можно убрать это условие
)

filtered_meta = meta[mask].copy()
print(f"\n=== После фильтра ===")
print(f"Партий: {len(filtered_meta)}")
print(f"Средний ЭЛО белых: {filtered_meta['white_elo'].mean():.0f}")
print(f"Средний ЭЛО чёрных: {filtered_meta['black_elo'].mean():.0f}")

# Если партий мало, пробуем с rapid
if len(filtered_meta) < 50000:
    print("\n=== Добавляем rapid ===")
    mask_with_rapid = (
        (meta["mode"].isin(["classical", "rapid"]))
        & (meta["white_elo"] >= 1800)
        & (meta["black_elo"] >= 1800)
    )
    filtered_meta = meta[mask_with_rapid].copy()
    print(f"Партий: {len(filtered_meta)}")

# Сохраняем разрешённые game_id
allowed_ids = set(filtered_meta["game_id"])

# Читаем ходы и фильтруем
print("\n=== Фильтрация ходов ===")
moves = pd.read_csv(MOVES, usecols=["game_id", "move_no", "move"])
moves = moves[moves["game_id"].isin(allowed_ids)]
print(f"Строк ходов после фильтра: {len(moves)}")

# Сортируем и склеиваем в партии
moves = moves.sort_values(["game_id", "move_no"], kind="mergesort")
games = (
    moves.groupby("game_id", sort=False)["move"]
         .apply(lambda seq: " ".join(seq))
         .tolist()
)

# Фильтр по длине: минимум 24 полухода (12 полных ходов)
games = [g for g in games if len(g.split()) >= 24]
print(f"Партий в датасете: {len(games)}")

# Сохраняем
OUT = Path("data/processed/games.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(games, ensure_ascii=False), encoding="utf-8")
print(f"Сохранено в {OUT}")