# Steel Defect Detection - MLOps Project

![CI](https://github.com/AntonAdonin/steel-defect-detection/workflows/CI/badge.svg)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Система детекции и классификации дефектов на стальных пластинах с использованием ансамбля YOLO + EfficientNet, развёрнутая через MLflow Serving.

## 🎯 Описание проекта

Проект решает задачу автоматической детекции и классификации дефектов на поверхности стальных пластин (Severstal Steel Defect Detection) для помощи инженерам в улучшении производственного процесса.

### Архитектура

```
┌─────────────┐         ┌──────────────┐         ┌─────────────────┐
│   Image     │────────>│  FastAPI     │────────>│  MLflow Model   │
│   Input     │         │  Backend     │         │                 │
└─────────────┘         │              │         │  ┌───────────┐  │
                        │  ┌────────┐  │         │  │   YOLO    │  │
                        │  │ MLflow │  │         │  │ Detector  │  │
                        │  │Tracking│  │         │  └─────┬─────┘  │
                        │  └────────┘  │         │        │         │
                        └──────────────┘         │  ┌─────▼─────┐  │
                                                 │  │EfficientNet│  │
                                                 │  │Classifier │  │
                                                 │  └───────────┘  │
                                                 └─────────────────┘
        MLflow Tracking + Model Registry + Serving
```

### Основные компоненты

1. **YOLO11s Detector** - детекция дефектов на изображении (сегментация)
2. **EfficientNet B0 Classifier** - классификация типов дефектов (4 класса)
3. **MLflow Model Registry** - версионирование и управление моделями
4. **FastAPI Backend** - REST API для инференса
5. **DVC** - версионирование данных
6. **GitHub Actions CI** - автоматические проверки кода

## 🚀 Быстрый старт

### Установка

```bash
# Установить UV (если еще не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Клонировать репозиторий
git clone https://github.com/AntonAdonin/steel-defect-detection.git
cd steel-defect-detection

# Установить зависимости
uv sync --all-extras

# Настроить Kaggle API (для загрузки данных)
# Скачайте kaggle.json с https://www.kaggle.com/settings
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Если нет доуступа к соревнованию
curl -L "https://drive.usercontent.google.com/download?id=1zQ2VKA6ng5M_oXFd6JtM_r6keebwBtfK&export=download&authuser=0&confirm=t&uuid=b105bbe9-bd14-4d50-ba40-9846b8be2e31&at=ANTm3cwd74EzDdpYsCve7IAvOP40%3A1766528697689" -o data.zip
unzip data.zip
rm -rf __MACOSX
rm -rf data.zip
```

### Полный пайплайн

```bash
# 1. Скачать датасет
uv run python commands.py download_data

# 2. Подготовить данные
# 2.1. При загрузки датасета через kaggle
uv run python commands.py prepare_yolo_kaggle
uv run python commands.py prepare_classification_kaggle

# 2.2. При загрузки датасета через curl
uv run python commands.py prepare_yolo
uv run python commands.py prepare_classification

# 3. Обучить модели
uv run python commands.py train_yolo
uv run python commands.py train_efficientnet

# 4. Экспортировать модели
uv run python commands.py export_yolo
uv run python commands.py export_efficientnet

# 5. Зарегистрировать в MLflow
uv run python commands.py register_model

# 6. Запустить MLflow UI (опционально)
uv run python commands.py start_mlflow

# 7. Запустить API
uv run python commands.py start_api

# 8. Тестировать API
uv run python commands.py test_api
```

**API доступен**: http://localhost:8000/docs
**MLflow UI**: http://localhost:5000

## 📁 Структура проекта

```
steel-defect-detection/
├── .github/workflows/            # GitHub Actions CI
│   ├── ci.yml                   # Линтинг, тесты, security
│   └── pre-commit.yml           # Pre-commit хуки
│
├── steel_defect_detection/      # Основной пакет
│   ├── data_processing/         # Работа с данными
│   │   ├── download.py          # Загрузка с Kaggle
│   │   ├── prepare_yolo_dataset.py
│   │   └── prepare_classification_dataset.py
│   ├── models/                  # Определения моделей
│   │   ├── yolo_detector.py
│   │   └── efficientnet_classifier.py
│   ├── training/                # Обучение
│   │   ├── efficientnet_module.py  # Lightning Module
│   │   ├── train_yolo.py
│   │   └── train_efficientnet.py
│   ├── export/                  # Экспорт в ONNX/PT
│   │   ├── export_yolo.py
│   │   └── export_efficientnet.py
│   ├── inference/               # Инференс
│   │   └── mlflow_model.py      # MLflow pyfunc wrapper
│   └── utils/                   # Утилиты
│
├── tests/                       # Тесты
│   ├── test_lightning_module.py
│   ├── test_mlflow_model.py
│   └── test_api.py
│
├── configs/                     # Hydra конфигурации
│   ├── data/default.yaml
│   ├── model/yolo_efficientnet.yaml
│   ├── train/default.yaml
│   ├── export/default.yaml
│   └── inference/default.yaml
│
├── data/                        # Данные (DVC tracked)
│   ├── raw/                     # Kaggle датасет
│   ├── yolo_dataset/            # YOLO format
│   └── classification_dataset/  # ImageFolder format
│
├── models/                      # Обученные модели
├── mlruns/                      # MLflow artifacts
├── commands.py                  # CLI для всех операций
├── mlflow_api.py               # FastAPI backend
├── pyproject.toml              # Зависимости и настройки
│
├── QUICKSTART.md               # Быстрый старт
```

## 🛠 CLI команды (commands.py)

### Данные

```bash
# Скачать датасет с Kaggle
uv run python commands.py download_data

# Подготовить YOLO датасет (RLE -> polygon)
uv run python commands.py prepare_yolo

# Подготовить classification датасет (crop defects)
uv run python commands.py prepare_classification
```

### Обучение

```bash
# Обучить YOLO (с MLflow tracking)
uv run python commands.py train_yolo

# Обучить EfficientNet (PyTorch Lightning)
uv run python commands.py train_efficientnet
```

### Экспорт

```bash
# Экспортировать YOLO в .pt
uv run python commands.py export_yolo

# Экспортировать EfficientNet в .pth
uv run python commands.py export_efficientnet
```

### MLflow

```bash
# Зарегистрировать ансамбль в Model Registry
uv run python commands.py register_model

# Запустить MLflow UI
uv run python commands.py start_mlflow

# Запустить FastAPI backend
uv run python commands.py start_api --port 8000
```

### Тестирование

```bash
# Тестировать API endpoints
uv run python commands.py test_api

# Запустить pytest
uv run python commands.py test

# Линтинг (ruff, black, isort)
uv run python commands.py lint

# Форматирование
uv run python commands.py format
```

### DVC

```bash
# Добавить данные в DVC
uv run python commands.py dvc_add_data

# Скачать данные из remote
uv run python commands.py dvc_pull

# Загрузить данные в remote
uv run python commands.py dvc_push

# Проверить статус
uv run python commands.py dvc_status
```

## 🎨 Качество кода

### Инструменты

- **Ruff** - быстрый линтер (замена flake8, isort)
- **Black** - форматирование кода
- **isort** - сортировка импортов
- **mypy** - проверка типов
- **pytest** - тестирование
- **pytest-cov** - покрытие тестами
- **bandit** - security scanner
- **radon** - анализ сложности
- **pre-commit** - автоматические проверки

### CI/CD

GitHub Actions автоматически проверяет:

- ✅ Линтинг (ruff, black, isort)
- ✅ Типизация (mypy)
- ✅ Тесты (pytest)
- ✅ Безопасность (bandit)
- ✅ Покрытие кода (coverage)
- ✅ Сложность кода (radon)

```bash
# Локально запустить проверки
uv run python commands.py lint
uv run python commands.py format
uv run python commands.py test

# Pre-commit хуки
uv run pre-commit install
uv run pre-commit run --all-files
```

## 📊 API Endpoints

### Мониторинг

- `GET /` - Информация об API
- `GET /health` - Проверка здоровья
- `GET /model/info` - Информация о модели

### Инференс

- `POST /predict/image` - Предсказание (multipart/form-data)

  ```bash
  curl -X POST http://localhost:8000/predict/image \
    -F "file=@image.jpg"
  ```

- `POST /predict/base64` - Предсказание (base64 JSON)
  ```bash
  curl -X POST http://localhost:8000/predict/base64 \
    -H "Content-Type: application/json" \
    -d '{"image_base64": "base64_string_here"}'
  ```

**Swagger UI**: http://localhost:8000/docs
**ReDoc**: http://localhost:8000/redoc

## 🧪 Тестирование

```bash
# Все тесты
uv run pytest tests/ -v

# С покрытием
uv run pytest tests/ --cov=steel_defect_detection --cov-report=html

# Конкретный тест
uv run pytest tests/test_api.py::test_health_endpoint -v

# Через commands.py
uv run python commands.py test
```

## 📈 Метрики и эксперименты

### MLflow Tracking

Все эксперименты логируются в MLflow:

```python
# YOLO training
- Metrics: box_loss, seg_loss, cls_loss, mAP50, mAP50-95
- Plots: confusion matrix, F1 curve, PR curve

# EfficientNet training
- Metrics: train_loss, val_loss, val_acc, val_f1, val_precision, val_recall
- Params: lr, batch_size, optimizer, epochs
```

Просмотр: http://localhost:5000

### Model Registry

Модели версионируются в MLflow Model Registry:

```bash
# Текущая версия
Model: steel_defect_ensemble
Version: 1
Stage: Staging
```

## 🗂 Управление данными (DVC)

```bash
# Инициализация (уже выполнено)
uv run dvc init

# Настроить remote storage
uv run dvc remote add -d myremote s3://my-bucket/dvcstore

# Добавить данные
uv run python commands.py dvc_add_data

# Коммит в Git
git add data.dvc .gitignore
git commit -m "Track data with DVC"

# Загрузить в remote
uv run python commands.py dvc_push

# Скачать на новой машине
git clone <repo>
uv run python commands.py dvc_pull
```

## 🐳 Docker (опционально)

Проект использует UV для управления зависимостями. Docker не обязателен, но можно использовать для deployment:

```bash
# Создать образ для API
docker build -t steel-defect-api .

# Запустить контейнер
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlruns:/app/mlruns \
  steel-defect-api
```

## 📚 Документация

- [QUICKSTART.md](QUICKSTART.md) - Быстрый старт за 5 минут

## 🔧 Требования

- Python 3.10+
- UV package manager
- Kaggle API (для загрузки данных)
- 8GB+ RAM
- GPU рекомендуется для обучения
