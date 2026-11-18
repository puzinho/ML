# финальная оценка модели на тестовом сплите
import argparse
import json
from pathlib import Path

import torch

from train import ChessLSTM, DEFAULT_CONFIG


def main():
    # разбор аргументов запуска
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    # выбор устройства вычислений
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # загрузка словаря и тензоров датасета
    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
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
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
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

    # вывод итоговых метрик
    print(f"test examples: {n}")
    print(f"test loss:     {loss_sum / n:.4f}")
    for k in hits:
        print(f"test Top-{k}:    {hits[k] / n:.4f}")


if __name__ == "__main__":
    main()