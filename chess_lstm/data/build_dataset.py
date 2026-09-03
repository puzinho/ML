# шаг 2: токенизация ходов, сборка датасета и загрузка в clearml
# скрипт читает games.json, строит словарь ходов, кодирует партии числами,
# нарезает примеры для lstm, сохраняет vocab.json и dataset.pt,
# затем создаёт датасет в clearml и загружает файлы на сервер

import json
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
CONTEXT_SIZE = 12
MIN_FREQ = 2
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

# настройки clearml
CLEARML_PROJECT = "Chess_LSTM"
CLEARML_DATASET_NAME = "Chess UCI Dataset"


def build_dataset():
    # читаем партии в формате uci из шага 1
    if not GAMES_PATH.exists():
        raise FileNotFoundError(f"не найден {GAMES_PATH}")

    with open(GAMES_PATH, "r", encoding="utf-8") as f:
        games = json.load(f)
    print(f"Загружено партий: {len(games)}")

    # считаем частоту каждого хода
    counter = Counter()
    for game in games:
        counter.update(game)

    # строим словарь: каждому частому ходу свой номер
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for move, freq in counter.items():
        if freq >= MIN_FREQ:
            vocab[move] = len(vocab)
    print(f"Размер словаря: {len(vocab)}")

    # заменяем ходы на их номера
    unk_id = vocab["<UNK>"]
    encoded = [[vocab.get(m, unk_id) for m in game] for game in games]

    # нарезаем партии на примеры: контекст и следующий ход
    X, y = [], []
    for game in tqdm(encoded, desc="Нарезка примеров"):
        if len(game) <= CONTEXT_SIZE:
            continue
        for i in range(CONTEXT_SIZE, len(game)):
            X.append(game[i - CONTEXT_SIZE:i])
            y.append(game[i])

    if len(X) == 0:
        raise ValueError("Ошибка, пустой файл Games.json")

    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    print(f"Создано примеров: {len(X)}")

    # делим на train, val и test
    total = len(X)
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "X_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "X_test": X[val_end:],
        "y_test": y[val_end:],
    }

    # сохраняем словарь и тензоры локально
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
            "train": len(splits["X_train"]),
            "val": len(splits["X_val"]),
            "test": len(splits["X_test"]),
        },
        "top_moves": counter.most_common(10),
    }


def upload_to_clearml(stats):
    # создаём датасет в clearml
    dataset = Dataset.create(
        dataset_project=CLEARML_PROJECT,
        dataset_name=CLEARML_DATASET_NAME,
    )
    dataset.add_tags(["chess", "uci", "lstm", "kaggle"])

    # загружаем всю папку с обработанными файлами
    dataset.add_files(path=str(PROCESSED_DIR))

    # пишем метаданные, чтобы статистику было видно в веб интерфейсе
    logger = dataset.get_logger()
    logger.report_text(
        f"games: {stats['n_games']}, vocab: {stats['vocab_size']}, "
        f"samples: {stats['n_samples']}, context: {CONTEXT_SIZE}"
    )

    splits_df = pd.DataFrame(list(stats["splits"].items()), columns=["split", "samples"])
    logger.report_table(title="Dataset Splits", series="Summary", table_plot=splits_df)

    top_df = pd.DataFrame(stats["top_moves"], columns=["move", "count"])
    logger.report_table(title="Top 10 Moves", series="Frequency", table_plot=top_df)

    # отправляем файлы на сервер и закрываем версию датасета
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