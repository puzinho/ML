# Интеллектуальный анализ данных и ML: Настройка окружения

## 0. Установка Git

Для работы с репозиторием вам понадобится установленный Git.

### Ubuntu/Debian

```
sudo apt-get update
sudo apt-get install git
```

### macOS

```
brew install git
```

### Windows

Скачайте и установите Git с официального сайта: [https://git-scm.com/](https://git-scm.com/)

## 1. Клонирование репозитория

```bash
git clone https://github.com/Eponeshnikov/Data-science-and-ML.git
cd Data-science-and-ML
```

## 2. Установка редактора кода

Для работы с проектом вам понадобится один из следующих редакторов:

### VS Code
Скачайте и установите Visual Studio Code с официального сайта: [https://code.visualstudio.com/](https://code.visualstudio.com/) 

После установки редактора выполните следующие шаги:
1. Откройте VS Code
2. Перейдите в раздел расширений: `Ctrl+Shift+X` или через меню `Вид -> Расширения`
3. Установите следующие необходимые расширения:
   - Python (Microsoft)
   - Python Debugger (Microsoft)
   - Python Environments (Microsoft)
   - Pylance (Microsoft)
   - Jupyter (Microsoft)

### PyCharm
Скачайте и установите PyCharm с официального сайта JetBrains: [https://www.jetbrains.com/pycharm/](https://www.jetbrains.com/pycharm/) 

Доступны две версии:
- **Community Edition** - бесплатная версия с базовыми функциями
- **Professional Edition** - платная версия с расширенными возможностями для веб- и data science разработки 

## 3. Установка uv

uv - это высокопроизводительный менеджер пакетов Python, написанный на Rust, который работает в 10-100 раз быстрее традиционных инструментов. 

Установить uv можно несколькими способами:

### Через pip
```bash
pip install uv
```

### Через установщик (рекомендуется)
- **Windows (PowerShell)**:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
- **macOS/Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Подробнее об установке можно узнать на официальном сайте: [https://astral.sh/blog/uv-unified-python-packaging](https://astral.sh/blog/uv-unified-python-packaging) 

## 4. Настройка окружения проекта

### 4.1. Проверка версии Python
В репозитории присутствует файл `.python-version`, который указывает требуемую версию Python для проекта.

### 4.2. Установка необходимой версии Python (при необходимости)
Если у вас не установлена требуемая версия Python, uv может установить её, также uv может загрузить ее автоматически при синхронизации (`uv sync`):

```bash
uv python install <версия_из_.python-version>
```

Например:
```bash
uv python install 3.12
```

uv автоматически загружает и устанавливает необходимые версии Python без предварительной настройки. 


### 4.3. Установка зависимостей
Установите все необходимые пакеты из файла `uv.lock`:

```bash
uv sync
```

Это обеспечит установку точных версий пакетов, указанных в lock-файле для воспроизводимости проекта. 

## 5. Запуск .ipynb файлов

Для работы с Jupyter Notebook файлами доступны следующие способы:
1. Использование встроенной поддержки в VSCode/PyCharm
2. Запуск веб-интерфейса через терминал:

```bash
jupyter-lab
```

## Дополнительная информация

- uv может управлять несколькими версиями Python одновременно, что удобно для работы с разными проектами. 
- Инструмент автоматически обнаруживает установленные версии Python и может устанавливать недостающие версии по требованию. 
- Подробную документацию по uv можно найти в официальных материалах: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/) 