# финальная оценка модели на тестовом сплите
import argparse
import json
from pathlib import Path

import torch

from train import ChessLSTM, DEFAULT_CONFIG

HERE = Path(__file__).parent


def main():
    # разбор аргументов запуска
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data" / "processed" / "dataset.pt")
    ap.add_argument("--vocab", type=Path, default=HERE / "data" / "processed" / "vocab.json")
    ap.add_argument("--weights", type=Path, default=HERE / "models" / "chess_lstm_best.pth")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    # выбор устройства вычислений
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # загрузка словаря ходов
    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    # контроль размера словаря: отсев символьной бракованной версии
    if len(vocab) < 100:
        raise ValueError(f"словарь из {len(vocab)} токенов: требуется словарь ходов версии v7")
    # загрузка тензоров датасета
    data = torch.load(args.data, map_location="cpu", weights_only=False)

    # инициализация архитектуры по конфигурации обучения
    model = ChessLSTM(
        vocab_size=len(vocab),
        emb_dim=DEFAULT_CONFIG["emb_dim"],
        hidden_size=DEFAULT_CONFIG["hidden_size"],
        num_layers=DEFAULT_CONFIG["num_layers"],
        dropout=DEFAULT_CONFIG["dropout"],
    ).to(device)
    # загрузка весов лучшего чекпоинта
    state = torch.load(args.weights, map_location=device, weights_only=True)
    # контроль согласованности словаря и весов: число классов должно совпадать
    expected = state["embedding.weight"].shape[0]
    if expected != len(vocab):
        raise ValueError(
            f"веса обучены на словаре из {expected} токенов, передан словарь из {len(vocab)}: "
            f"возьми vocab.json и dataset.pt из той же версии датасета, что и веса"
        )
    model.load_state_dict(state)
    model.eval()

    # выделение тестового сплита
    X, y = data["X_test"], data["y_test"]
    criterion = torch.nn.CrossEntropyLoss()
    hits = {1: 0, 3: 0, 5: 0}
    loss_sum, n = 0.0, 0

    # пакетная оценка без построения градиентов
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            xb = X[i:i + args.batch].to(device).long()
            yb = y[i:i + args.batch].to(device).long()
            logits = model(xb)
            # накопление взвешенной потери
            loss_sum += criterion(logits, yb).item() * len(yb)
            n += len(yb)
            # накопление попаданий в топ-k
            for k in hits:
                top = logits.topk(k, dim=-1).indices
                hits[k] += (top == yb.unsqueeze(1)).any(dim=1).sum().item()
            # промежуточный вывод прогресса
            if (i // args.batch) % 500 == 0:
                print(f"обработано {n} из {len(X)} примеров", flush=True)

    # вывод итоговых метрик
    print(f"test examples: {n}")
    print(f"test loss:     {loss_sum / n:.4f}")
    for k in hits:
        print(f"test Top-{k}:    {hits[k] / n:.4f}")


if __name__ == "__main__":
    main()