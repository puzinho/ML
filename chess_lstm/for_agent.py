from clearml import Task

task = Task.create(
    project_name="Chess_LSTM",
    task_name="LSTM V8 (Colab)",
    repo="https://github.com/puzinho/ML.git",
    branch="main",
    script="chess_lstm/train.py",
)
Task.enqueue(task, queue_name="colab_queue")
print(task.id)