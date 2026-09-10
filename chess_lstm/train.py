# chess_lstm/train.py
import torch
import json
from torch.utils.data import DataLoader, TensorDataset
from clearml import Task

# Импортируем нашу модель
from models.chess_lstm import ChessLSTM

# 1. Инициализация ClearML
task = Task.init(
    project_name="Chess_LSTM",
    task_name="Baseline v1",
    output_uri=True  # сохраняет артефакты на сервер ClearML
)

# 2. Загрузка датасета
dataset = torch.load("data/processed/dataset.pt", weights_only=False)

# Подготовка DataLoader
train_data = TensorDataset(dataset["X_train"], dataset["y_train"])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 3. Создание модели
with open("data/processed/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)  # берём размер словаря
vocab_size = len(vocab)
model = ChessLSTM(vocab_size=vocab_size)

# 4. Функция ошибки и оптимизатор
criterion = torch.nn.CrossEntropyLoss()  # ключевая метрика для классификации
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Логирование гиперпараметров в ClearML
task.connect({
    "vocab_size": vocab_size,
    "batch_size": 64,
    "lr": 0.001,
    "emb_dim": 64,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.1
})

# 6. Обучение
for epoch in range(3):  # 3 эпохи для теста
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Эпоха {epoch+1} | Loss: {avg_loss:.4f}")
    task.logger.report_scalar("Loss", "train", value=avg_loss, iteration=epoch)

# 7. Сохранение модели
torch.save(model.state_dict(), "models/chess_lstm.pth")
task.upload_artifact("model", "models/chess_lstm.pth")
print("\n✅ Модель сохранена и залита в ClearML!")