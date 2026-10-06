import torch
from torch.utils.data import DataLoader, TensorDataset
from clearml import Task

from data_loader import CHECKPOINT_DIR, ensure_data, load_dataset, load_vocab
from models.chess_lstm import ChessLSTM

DEFAULT_CONFIG = {
    "epochs": 30,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "emb_dim": 128,
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.2,
    "patience": 3,
}


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    top_k = logits.topk(k, dim=1).indices
    return (top_k == targets.unsqueeze(1)).any(dim=1).float().mean().item()


def markov_baseline_top5(y_train, y_val, vocab_size):
    """Бейзлайн: для каждого последнего хода - топ-5 самых частых ответов на него в трейне."""
    from collections import defaultdict
    # Считаем частоту пар (последний ход -> следующий ход)
    transitions = defaultdict(lambda: torch.zeros(vocab_size))
    for prev, nxt in zip(y_train[:-1], y_train[1:]):
        transitions[prev.item()][nxt.item()] += 1
    
    # Для каждого val-примера: берём его предпоследний ход, находим топ-5 ответов на него
    hits = 0
    for i in range(len(y_val)):
        if i == 0:
            continue  # у первого val-примера нет предыдущего хода из трейна
        last_move = y_val[i-1].item()
        top5 = transitions[last_move].topk(5).indices
        if y_val[i].item() in top5.tolist():
            hits += 1
    return hits / len(y_val)


def make_loaders(dataset: dict, batch_size: int):
    train = TensorDataset(dataset["X_train"], dataset["y_train"])
    val = TensorDataset(dataset["X_val"], dataset["y_val"])
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
    )


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device).long(), y.to(device).long()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        batches += 1
    return total / batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, batches = 0.0, 0
    hits = {1: 0.0, 3: 0.0, 5: 0.0}
    for x, y in loader:
        x, y = x.to(device).long(), y.to(device).long()
        logits = model(x)
        total += criterion(logits, y).item()
        batches += 1
        for k in hits:
            hits[k] += topk_accuracy(logits, y, k)
    return total / batches, {k: v / batches for k, v in hits.items()}


def main():
    ensure_data()
    
    task = Task.init(project_name="Chess_LSTM", task_name="train")
    config = task.connect(DEFAULT_CONFIG)  # гиперпараметры видны и редактируемы в UI
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    vocab_size = len(load_vocab())
    dataset = load_dataset()
    train_loader, val_loader = make_loaders(dataset, config["batch_size"])

    baseline = markov_baseline_top5(dataset["y_train"], dataset["y_val"], vocab_size)
    print(f"[train] марковский бейзлайн Top-5: {baseline:.4f}")
    task.logger.report_scalar("Baseline", "Top-5", value=baseline, iteration=0)

    model = ChessLSTM(
        vocab_size=vocab_size,
        emb_dim=config["emb_dim"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_DIR / "chess_lstm_best.pth"
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, acc = evaluate(model, val_loader, criterion, device)

        task.logger.report_scalar("Loss", "train", value=train_loss, iteration=epoch)
        task.logger.report_scalar("Loss", "val", value=val_loss, iteration=epoch)
        for k, value in acc.items():
            task.logger.report_scalar("Accuracy", f"Top-{k}", value=value, iteration=epoch)
        print(
            f"[train] epoch {epoch + 1:2d} | loss {train_loss:.4f} | val {val_loss:.4f} | "
            f"top1 {acc[1]:.3f} top3 {acc[3]:.3f} top5 {acc[5]:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
            print(f"[train]   новый лучший результат, чекпоинт сохранён")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                print(f"[train] early stopping на эпохе {epoch + 1}, best val {best_val_loss:.4f}")
                break

    print(f"[train] готово, best val loss: {best_val_loss:.4f}")
    task.upload_artifact("model_weights", str(checkpoint))


if __name__ == "__main__":
    main()