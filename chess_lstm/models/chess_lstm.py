# chess_lstm/models/chess_lstm.py
import torch
import torch.nn as nn


class ChessLSTM(nn.Module):
    """
    Модель для предсказания следующего хода в шахматах на основе истории полуходов.

    Вход: последовательность из CONTEXT_SIZE ходов (номеров из словаря).
    Выход: вероятности следующего хода для всех возможных вариантов.

    Пример:
        Вход: [10, 22, 35, 41, 57, 62, 99, 112, 150, 167, 180, 200]
        Выход: вероятности для 1800 ходов (словарь), например, ход 150 имеет вероятность 0.7
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Параметры модели:
        - vocab_size: размер словаря ходов (из vocab.json)
        - emb_dim: размер эмбеддинга (как много чисел описывает один ход)
        - hidden_size: размер "памяти" LSTM (сколько информации хранит модель)
        - num_layers: глубина сети
        - dropout: защита от переобучения (случайно отключает нейроны)
        """
        super().__init__()
        self.vocab_size = vocab_size

        # 1. Слой эмбеддингов: превращает номер хода в вектор признаков
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=0  # <PAD> как нулевой индекс, чтобы LSTM игнорировал его при обучении
        )

        # 2. LSTM: читает последовательность ходов
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # dropout только для >1 слоя
        )

        # 3. Линейный слой: преобразует состояние LSTM в оценки для всех ходов
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.

        Аргументы:
            x: тензор формы (batch_size, CONTEXT_SIZE)
               где каждый элемент — номер хода из словаря

        Возвращает:
            Логиты (сырые оценки) для каждого возможного следующего хода.
            Форма: (batch_size, vocab_size)
        """
        # Шаг 1: эмбеддинги (номера ходов → векторы)
        x = self.embedding(x)  # [batch, context] → [batch, context, emb_dim]

        # Шаг 2: LSTM обрабатывает последовательность
        out, _ = self.lstm(x)  # [batch, context, emb_dim] → [batch, context, hidden]

        # Шаг 3: берём только последнее состояние (все предыдущие ходы учтены)
        last_hidden = out[:, -1, :]  # [batch, hidden]

        # Шаг 4: преобразуем в оценки для всех ходов
        logits = self.fc(last_hidden)  # [batch, hidden] → [batch, vocab_size]

        return logits


# Самопроверка (запускается только при прямом запуске файла)
if __name__ == "__main__":
    # 1. Проверяем размерности
    vocab_size = 1850  # Примерно таков размер словаря из build_dataset.py
    batch_size = 4
    context_size = 12

    # Случайный батч: 4 примера по 12 полуходов каждый
    dummy_input = torch.randint(0, vocab_size, (batch_size, context_size))

    # Создаём модель
    model = ChessLSTM(vocab_size=vocab_size)

    # Прогоняем через модель
    output = model(dummy_input)

    # Проверяем формы
    assert output.shape == (batch_size, vocab_size), \
        f"Ошибка формы: ожидалось ({batch_size}, {vocab_size}), получено {output.shape}"

    # Проверяем, что softmax даёт вероятности (сумма = 1)
    probs = torch.softmax(output, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size)), \
        "Сумма вероятностей не равна 1"

    print("✅ Модель работает корректно!")
    print(f"Форма входа: {dummy_input.shape} (batch, context_size)")
    print(f"Форма выхода: {output.shape} (batch, vocab_size)")
    print(f"Пример вероятностей: {probs[0, :5].detach().numpy()}")