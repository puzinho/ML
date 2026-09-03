# data/csv_parser.py
import json
from pathlib import Path

import chess
import pandas as pd
from tqdm import tqdm

CSV_PATH = Path("C:/Users/neddy/Machine Learning/ml/chess_lstm/data/raw/chess_games.csv")   # <- имя твоего файла
PROCESSED_PATH = Path("C:/Users/neddy/Machine Learning/ml/chess_lstm/data/processed/games.json")
MAX_GAMES = 20000     
MIN_MOVES = 8         
MIN_RATING = 1500   
NROWS = 100000         


def san_to_uci(san_moves):
    """Конвертирует SAN-ходы в UCI с валидацией на реальной доске."""
    board = chess.Board()
    uci = []
    for san in san_moves:
        try:
            move = board.push_san(san)
        except ValueError:      # нелегальный/битый ход -> брак
            return None
        uci.append(move.uci())
    return uci


def main():
    df = pd.read_csv(CSV_PATH, nrows=NROWS)
    print("Колонки файла:", list(df.columns))

    # Ищем СТРОКОВУЮ колонку с текстом ходов (turns здесь числовая, её пропускаем)
    candidates = ["moves", "turns", "Moves", "pgn"]
    moves_col = None
    for c in candidates:
        if c in df.columns and df[c].dtype == object:
            moves_col = c
            break
    if moves_col is None:
        raise ValueError("Не нашёл колонку с ходами: " + str(list(df.columns)))

    # Фильтр качества (если в датасете есть рейтинги)
    if {"white_rating", "black_rating"}.issubset(df.columns):
        df = df[(df["white_rating"] >= MIN_RATING) & (df["black_rating"] >= MIN_RATING)]
        print(f"Партий после фильтра рейтинга: {len(df)}")

    games = []
    for turns in tqdm(df[moves_col], desc="Парсинг партий"):
        # убираем токены вида "1." и "12..." (номера ходов), если они есть
        san_moves = [t for t in str(turns).split() if not t.endswith(".")]
        if len(san_moves) < MIN_MOVES:
            continue
        uci = san_to_uci(san_moves)
        if uci is not None:
            games.append(uci)
        if len(games) >= MAX_GAMES:
            break

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(games, f, ensure_ascii=False)

    print(f"Сохранено партий: {len(games)}")
    if games:
        print("Пример:", games[0][:10])


if __name__ == "__main__":
    main()