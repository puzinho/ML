# интерактивное предсказание следующего хода по истории партии
import argparse
import json
from pathlib import Path

import torch

from train import ChessLSTM, DEFAULT_CONFIG

HERE = Path(__file__).parent


def load_model(vocab_path: Path, weights_path: Path, device: torch.device):
    # загрузка словаря ходов
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    # контроль размера словаря: отсев символьной бракованной версии
    if len(vocab) < 100:
        raise ValueError(f"словарь из {len(vocab)} токенов: требуется словарь ходов версии v7")
    # инициализация архитектуры по конфигурации обучения
    model = ChessLSTM(
        vocab_size=len(vocab),
        emb_dim=DEFAULT_CONFIG["emb_dim"],
        hidden_size=DEFAULT_CONFIG["hidden_size"],
        num_layers=DEFAULT_CONFIG["num_layers"],
        dropout=DEFAULT_CONFIG["dropout"],
    ).to(device)
    # загрузка весов лучшего чекпоинта
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    # перевод в режим оценки: отключение dropout
    model.eval()
    return model, vocab


def encode(history, vocab, context):
    # кодирование ходов в индексы словаря, неизвестные ходы в <UNK>
    ids = [vocab.get(m, vocab["<UNK>"]) for m in history]
    unknown = [m for m in history if m not in vocab]
    # отбор последнего окна контекста
    ids = ids[-context:]
    # дополнение паддингом слева при нехватке истории
    if len(ids) < context:
        ids = [vocab["<PAD>"]] * (context - len(ids)) + ids
    return ids, unknown


def top5(model, ids, idx2move, device):
    # прямой проход модели
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1)[0]
    # отбор пяти наибольших вероятностей
    top = probs.topk(5)
    # обратное отображение индексов в ходы
    return [(idx2move[int(i)], float(p)) for p, i in zip(top.values, top.indices)]


def main():
    # разбор аргументов запуска
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=Path, default=HERE / "data" / "processed" / "vocab.json")
    ap.add_argument("--weights", type=Path, default=HERE / "models" / "best.pt")
    ap.add_argument("--context", type=int, default=12)
    args = ap.parse_args()

    # выбор устройства вычислений
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = load_model(args.vocab, args.weights, device)
    # обратный словарь для расшифровки предсказаний
    idx2move = {i: m for m, i in vocab.items()}
    history = []
    print("ходы через пробел - добавить | p - предсказать | r - сброс | q - выход")

    # цикл интерактивного ввода
    while True:
        line = input(f"[{len(history)} pli] > ").strip()
        # обработка команды выхода
        if line in ("q", "quit", "exit"):
            break
        # обработка команды сброса истории
        if line in ("r", "reset"):
            history = []
            continue
        # обработка команды предсказания
        if line in ("p", ""):
            if not history:
                print("история пуста")
                continue
            ids, unknown = encode(history, vocab, args.context)
            # предупреждение о неизвестных ходах
            if unknown:
                print("неизвестные ходы заменены на <UNK>:", unknown)
            # предупреждение о короткой истории
            if len(history) < args.context:
                print(f"истории меньше {args.context} pli: дополнение <PAD>, качество ниже")
            candidates = top5(model, ids, idx2move, device)
            # вывод ответа модели и списка кандидатов
            answer, p = candidates[0]
            print(f"ответ модели: {answer} ({p:.2%})")
            for i, (m, q) in enumerate(candidates, 1):
                print(f"  {i}. {m:6s} {q:6.2%}")
            continue
        # накопление истории партии
        history.extend(line.split())


if __name__ == "__main__":
    main()