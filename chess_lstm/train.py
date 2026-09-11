# chess_lstm/train.py
import torch
import json
import sklearn
from sklearn.metrics import top_k_accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from clearml import Task

# Импортируем нашу модель
from models.chess_lstm import ChessLSTM

# 1. Инициализация ClearML
task = Task.init(
    project_name="Chess_LSTM",
    task_name="LSTM v4",
    output_uri=True  # сохраняет артефакты на сервер ClearML
)

# 2. Загрузка датасета
dataset = torch.load("data/processed/dataset.pt", weights_only=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # вычисления на gpu, если доступно
print(f"устройство: {device}")
# Подготовка DataLoader
train_data = TensorDataset(dataset["X_train"], dataset["y_train"])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)


# создание baseline модели для сравнения
def calculate_baseline_topk():
    """Вычисляет Top-K Accuracy для частотной модели"""
    with open("data/processed/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    # Сортируем ходы по частоте
    sorted_moves = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    
    # Берём только реальные ходы (без <PAD> и <UNK>)
    sorted_moves = [move for move in sorted_moves if move[0] not in ["<PAD>", "<UNK>"]]
    
    # Создаём "предсказание" - топ-5 самых частых ходов
    top5_moves = [move[1] for move in sorted_moves[:5]]
    
    # Вычисляем, как часто правильный ход попадает в топ-5
    val_targets = dataset["y_val"].cpu().numpy()
    correct_in_top5 = sum(1 for target in val_targets if target in top5_moves)
    top5_baseline = correct_in_top5 / len(val_targets)
    
    return top5_baseline

baseline_top5 = calculate_baseline_topk()
task.logger.report_scalar("Baseline", "Top-5", value=baseline_top5, iteration=0)

# 3. Создание модели
with open("data/processed/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)  # берём размер словаря
vocab_size = len(vocab)
model = ChessLSTM(vocab_size=vocab_size).to(device)  # переносим модель на GPU, если доступно


def calculate_topk_accuracy(logits, targets, k=5): # вычисляем Top-K Accuracy для батча
    probs = torch.softmax(logits, dim=1)
    return top_k_accuracy_score(
        y_true=targets.cpu().numpy(),
        y_score=probs.cpu().numpy(),
        k=k,
        labels=list(range(vocab_size))
    )

# 4. Функция ошибки и оптимизатор
criterion = torch.nn.CrossEntropyLoss().to(device)  # ключевая метрика для классификации
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Логирование гиперпараметров в ClearML
task.connect({
    "vocab_size": vocab_size,
    "batch_size": 64,
    "lr": 0.001,
    "emb_dim": 64,
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.3
})

# 6. Обучение
for epoch in range(12):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    # Валидация
    model.eval()  # Переключаем модель в режим оценки
    val_loss = 0
    val_top1 = 0
    val_top3 = 0
    val_top5 = 0
    n_val_batches = 0

    # Подготовка DataLoader для валидации
    val_data = TensorDataset(dataset["X_val"], dataset["y_val"])
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    with torch.no_grad():  # Отключаем вычисление градиентов
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Считаем метрики
            top1 = calculate_topk_accuracy(logits, y_batch, k=1)
            top3 = calculate_topk_accuracy(logits, y_batch, k=3)
            top5 = calculate_topk_accuracy(logits, y_batch, k=5)
            
            val_loss += loss.item()
            val_top1 += top1
            val_top3 += top3
            val_top5 += top5
            n_val_batches += 1

    # Усредняем по батчам
    avg_val_loss = val_loss / n_val_batches
    avg_val_top1 = val_top1 / n_val_batches
    avg_val_top3 = val_top3 / n_val_batches
    avg_val_top5 = val_top5 / n_val_batches

    # Логируем в ClearML
    task.logger.report_scalar("Loss", "val", value=avg_val_loss, iteration=epoch)
    task.logger.report_scalar("Accuracy", "Top-1 (val)", value=avg_val_top1, iteration=epoch)
    task.logger.report_scalar("Accuracy", "Top-3 (val)", value=avg_val_top3, iteration=epoch)
    task.logger.report_scalar("Accuracy", "Top-5 (val)", value=avg_val_top5, iteration=epoch)

    print(f"Валидация | Loss: {avg_val_loss:.4f} | Top-1: {avg_val_top1:.4f} | Top-5: {avg_val_top5:.4f}")
    print(f"Эпоха {epoch+1} | Loss: {avg_loss:.4f}")
    task.logger.report_scalar("Loss", "train", value=avg_loss, iteration=epoch)


# 7. Сохранение модели
torch.save(model.state_dict(), "models/chess_lstm.pth")
task.upload_artifact("model", "models/chess_lstm.pth")
print("\nМодель сохранена и залита в ClearML")