# 🚀 Quick Start - Steel Defect Detection

## Минимальная последовательность для запуска

### 1. Проверка окружения

```bash
# Убедитесь что все зависимости установлены
uv sync

# Проверка работы CLI
uv run python commands.py
```

### 2. Если модели уже есть

```bash
# Запустить MLflow сервер (терминал 1)
uv run python commands.py start_mlflow

# Зарегистрировать модели в MLflow (терминал 2)
uv run python commands.py register_model

# Запустить API
uv run python commands.py start_api

# Тест API (терминал 3)
uv run python commands.py test_api
```

### 3. Если нужно обучить с нуля

```bash
# Терминал 1: MLflow сервер
uv run python commands.py start_mlflow

# Терминал 2: Подготовка и обучение
uv run python commands.py download_data
uv run python commands.py prepare_all_data
uv run python commands.py train_yolo
uv run python commands.py train_efficientnet

# Регистрация моделей
uv run python commands.py register_model

# Запуск API
uv run python commands.py start_api

# Терминал 3: Тест
uv run python commands.py test_api
```

## 📊 Ссылки после запуска

- **MLflow UI**: http://127.0.0.1:8080
- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

## ⚠️ Важные заметки

### Kaggle API

Если нет `~/.kaggle/kaggle.json`:

```bash
# Скачать с https://www.kaggle.com/settings/account
# Положить в ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Проблемы с MLflow

```bash
# Если модель не загружается в API
# Проверьте что MLflow сервер запущен
curl http://127.0.0.1:8080

# Проверьте Model Registry
# Откройте http://127.0.0.1:8080 → Models
```

### Структура файлов после обучения

```
models/
├── yolo_best.pt                      # YOLO веса
├── efficientnet_steel_defect.pth     # EfficientNet веса
└── checkpoints/                      # Lightning checkpoints

data/
├── raw/                              # Исходные данные
├── yolo_dataset/                     # YOLO датасет
└── classification_dataset/           # EfficientNet датасет
```

## 🔧 Полезные команды

```bash
# Список всех команд
uv run python commands.py

# Помощь по команде
uv run python commands.py train_yolo -- --help

# Форматирование кода
uv run python commands.py format

# Pre-commit
pre-commit run -a
```

## 📚 Документация

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Обучение моделей
- [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md) - MLflow Serving
- [README.md](README.md) - Общая информация
