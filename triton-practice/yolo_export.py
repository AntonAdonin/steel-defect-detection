from ultralytics import YOLO

# 1. Загружаем модель YOLO11s
model = YOLO('yolo11s.pt')  # скачает автоматически, если файла нет

# 2. Экспортируем в формат ONNX
model.export(
    format='onnx',       # формат экспорта
    opset=12,            # версия ONNX opset (можно 12–17)
    simplify=True,       # упрощает граф (через onnx-simplifier)
    dynamic=True          # включает динамические размеры входа
)

print("✅ Экспорт завершён! Файл будет сохранён в папке 'runs/export' или рядом с моделью.")
