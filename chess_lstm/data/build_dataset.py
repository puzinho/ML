# шаг 2: токенизация ходов, сборка датасета и загрузка в clearml
# скрипт читает games.json, строит словарь ходов, кодирует партии числами,
# нарезает примеры для lstm, сохраняет vocab.json и dataset.pt,
# затем создаёт датасет в clearml и загружает файлы на сервер

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from clearml import Dataset
from tqdm import tqdm

# привязываем пути к местоположению самого скрипта,
# чтобы запуск работал из любой папки
PROCESSED_DIR = Path(__file__).parent / "processed"
GAMES_PATH = PROCESSED_DIR / "games.json"
VOCAB_PATH = PROCESSED_DIR / "vocab.json"
DATASET_PATH = PROCESSED_DIR / "dataset.pt"

# параметры подготовки данных
CONTEXT_SIZE = 24          # сколько последних полуходов видит модель на входе
MIN_FREQ = 2               # ходы реже этого порога уходят в <UNK>: словарь не раздувается мусором
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SPLIT_SEED = 42            # фиксированное перемешивание: сплиты воспроизводимы между запусками
FLUSH_EVERY = 1_000_000    # примеров в буфере до превращения в тензор: ограничивает пик памяти

# настройки clearml
CLEARML_PROJECT = "Chess_LSTM"
# имя обязано совпадать с DATASET_NAME в data_loader.py, иначе обучающий код
# не найдёт новую версию и продолжит тянуть старый датасет
CLEARML_DATASET_NAME = "Chess UCI Dataset"


def tokenize_game(game):
    """Приводит партию к списку ходов независимо от формата хранения:
    строку режем по пробелам, список принимаем как есть."""
    if isinstance(game, str):
        return game.split()
    return list(game)


def build_vocab(games):
    """Считает частоты ходов по всем партиям и нумерует те, что проходят порог MIN_FREQ."""
    counter = Counter()
    for game in games:
        counter.update(tokenize_game(game))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for move, freq in counter.items():
        if freq >= MIN_FREQ:
            vocab[move] = len(vocab)
    return vocab, counter


def slice_games(games, vocab, desc):
    """Нарезает партии на пары (контекст из CONTEXT_SIZE ходов, следующий ход).
    Примеры не копятся списками до конца: каждые FLUSH_EVERY штук буфер
    превращается в компактный тензор int16, поэтому память не растёт с объёмом датасета."""
    unk_id = vocab["<UNK>"]
    chunks_x, chunks_y = [], []
    buf_x, buf_y = [], []

    for game in tqdm(games, desc=desc):
        ids = [vocab.get(m, unk_id) for m in tokenize_game(game)]
        if len(ids) <= CONTEXT_SIZE:
            continue
        for i in range(CONTEXT_SIZE, len(ids)):
            buf_x.append(ids[i - CONTEXT_SIZE:i])
            buf_y.append(ids[i])
        if len(buf_x) >= FLUSH_EVERY:
            chunks_x.append(torch.tensor(buf_x, dtype=torch.int16))
            chunks_y.append(torch.tensor(buf_y, dtype=torch.int16))
            buf_x, buf_y = [], []

    if buf_x:
        chunks_x.append(torch.tensor(buf_x, dtype=torch.int16))
        chunks_y.append(torch.tensor(buf_y, dtype=torch.int16))
    if not chunks_x:
        return torch.empty(0, CONTEXT_SIZE, dtype=torch.int16), torch.empty(0, dtype=torch.int16)
    return torch.cat(chunks_x), torch.cat(chunks_y)


def build_dataset():
    if not GAMES_PATH.exists():
        raise FileNotFoundError(f"не найден {GAMES_PATH}")

    with open(GAMES_PATH, "r", encoding="utf-8") as f:
        games = json.load(f)
    print(f"Загружено партий: {len(games)}")

    # проверка контракта данных: токен партии обязан быть UCI-ходом из 4-5 символов,
    # иначе мы снова едем на символах и узнаем об этом только по метрикам
    sample = tokenize_game(games[0])
    assert all(len(t) in (4, 5) for t in sample), "токены партии не похожи на UCI-ходы"

    vocab, counter = build_vocab(games)
    assert len(vocab) > 100, "словарь подозрительно мал: проверь токенизацию"
    assert len(vocab) < 32767, "словарь не помещается в int16"
    print(f"Размер словаря: {len(vocab)}")
    print("Пример токенов:", list(vocab.items())[2:7])

    # делим сначала партии, а не примеры: окна одной партии не должны
    # растекаться по разным сплитам, иначе val подглядывает в train
    order = list(range(len(games)))
    random.Random(SPLIT_SEED).shuffle(order)
    n = len(order)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_games = [games[i] for i in order[:train_end]]
    val_games = [games[i] for i in order[train_end:val_end]]
    test_games = [games[i] for i in order[val_end:]]

    X_train, y_train = slice_games(train_games, vocab, "Нарезка train")
    X_val, y_val = slice_games(val_games, vocab, "Нарезка val")
    X_test, y_test = slice_games(test_games, vocab, "Нарезка test")

    total = len(X_train) + len(X_val) + len(X_test)
    if total == 0:
        raise ValueError("не нарезано ни одного примера: проверь games.json")
    print(f"Создано примеров: {total} (train {len(X_train)}, val {len(X_val)}, test {len(X_test)})")

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    torch.save(splits, DATASET_PATH)
    print(f"Сохранено: {VOCAB_PATH}")
    print(f"Сохранено: {DATASET_PATH}")

    return {
        "n_games": len(games),
        "vocab_size": len(vocab),
        "n_samples": total,
        "splits": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "top_moves": counter.most_common(10),
    }


def upload_to_clearml(stats):
    dataset = Dataset.create(
        dataset_project=CLEARML_PROJECT,
        dataset_name=CLEARML_DATASET_NAME,
    )
    dataset.add_tags(["chess", "uci", "lstm", "high_elo"])

    # на сервер уходят только готовые артефакты: словарь и тензоры.
    # games.json остаётся локально как воспроизводимое сырьё
    dataset.add_files(path=str(VOCAB_PATH))
    dataset.add_files(path=str(DATASET_PATH))

    # статистика сборки текстом и таблицами, чтобы состав версии был виден в веб-интерфейсе
    logger = dataset.get_logger()
    logger.report_text(
        f"games: {stats['n_games']}, vocab: {stats['vocab_size']}, "
        f"samples: {stats['n_samples']}, context: {CONTEXT_SIZE}"
    )

    splits_df = pd.DataFrame(list(stats["splits"].items()), columns=["split", "samples"])
    logger.report_table(title="Dataset Splits", series="Summary", table_plot=splits_df)

    top_df = pd.DataFrame(stats["top_moves"], columns=["move", "count"])
    logger.report_table(title="Top 10 Moves", series="Frequency", table_plot=top_df)

    dataset.upload()
    dataset.finalize()

    print("Датасет загружен в ClearML")
    print(f"Dataset ID: {dataset.id}")
    print(f"Project: {dataset.project}")
    print(f"Name: {dataset.name}")


if __name__ == "__main__":
    stats = build_dataset()
    try:
        upload_to_clearml(stats)
    except Exception as e:
        print("Не удалось загрузить датасет в ClearML:", e)