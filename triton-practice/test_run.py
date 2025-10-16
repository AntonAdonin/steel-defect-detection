#!/usr/bin/env python3
"""
test_run.py

Отправляет три типа запросов в Triton:
- случайное изображение (mock)
- реальная фотография (из файла)
- "пустые" данные (нулевой тензор)

Каждый пример оформлен как отдельная функция и вызывается в main().
"""
import sys
import numpy as np
from PIL import Image
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from typing import Tuple, List

TRITON_URL = "localhost:8000"   # <-- поменяй если Triton на другом хосте/порте
MODEL_NAME = "yolo11s"
# Целевой размер для модели (обычно 640x640 для YOLO; подставь свой, если другой)
TARGET_H = 640
TARGET_W = 640
BATCH_SIZE = 1  # мы отправляем один элемент (Triton ожидает батч-ось при max_batch_size > 0)

def get_model_io(client: httpclient.InferenceServerClient, model_name: str) -> Tuple[str, List[str]]:
    """
    Получаем имена входа и выходов модели у Triton.
    Возвращает (input_name, [output_names...]).
    Если не удалось, возвращает дефолты ('images', ['output0']).
    """
    try:
        md = client.get_model_metadata(model_name=model_name)
        inputs = [inp.name for inp in md.inputs]
        outputs = [out.name for out in md.outputs]
        print(f"Model metadata inputs: {inputs}, outputs: {outputs}")
        # выбор первого входа и всех выходов
        input_name = inputs[0] if inputs else "images"
        return input_name, outputs or ["output0"]
    except Exception as e:
        print(f"Не удалось получить metadata у Triton (используются дефолтные имена). Причина: {e}")
        return "images", ["output0"]

def prepare_image_np(img: Image.Image, target_h: int, target_w: int) -> np.ndarray:
    """
    Ресайз, приведение в формат CHW и нормализация в float32.
    Возвращает массив формы (BATCH, C, H, W).
    """
    # Приводим изображение в RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Ресайз (с сохранением соотношения можно добавить letterbox; здесь простой ресайз)
    img = img.resize((target_w, target_h), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)  # H W C
    # Нормализация: в большинстве YOLO -- деление на 255.0
    arr /= 255.0
    # Переставляем в CHW
    arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)  # C H W
    # Добавляем батч-ось
    arr = np.expand_dims(arr, axis=0)  # 1 C H W
    return arr

def infer_random(client: httpclient.InferenceServerClient, model_name: str, input_name: str, output_names: List[str]):
    """
    Отправляет случайное изображение (mock) на модель.
    """
    print("\n--- infer_random ---")
    shape = (BATCH_SIZE, 3, TARGET_H, TARGET_W)
    image = np.random.rand(*shape).astype(np.float32)  # в диапазоне [0,1]
    try:
        inp = httpclient.InferInput(input_name, image.shape, "FP32")
        inp.set_data_from_numpy(image, binary_data=True)
        outputs = [httpclient.InferRequestedOutput(oname, binary_data=True) for oname in output_names]
        resp = client.infer(model_name=model_name, inputs=[inp], outputs=outputs)
        # печатаем все выходы
        for oname in output_names:
            try:
                out_np = resp.as_numpy(oname)
                print(f"Output '{oname}' shape: {None if out_np is None else out_np.shape}")
            except Exception as e:
                print(f"Не удалось получить выход {oname}: {e}")
    except InferenceServerException as e:
        print("Ошибка Triton:", e)
    except Exception as e:
        print("Ошибка:", e)

def infer_image(client: httpclient.InferenceServerClient, model_name: str, input_name: str, output_names: List[str], image_path: str):
    """
    Отправляет реальную фотографию (image_path) на модель.
    Если файла нет или не получилось прочитать -- печатает ошибку.
    """
    print("\n--- infer_image ---")
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Не удалось открыть изображение '{image_path}': {e}")
        return

    image_np = prepare_image_np(img, TARGET_H, TARGET_W)  # shape (1,3,H,W)
    try:
        inp = httpclient.InferInput(input_name, image_np.shape, "FP32")
        inp.set_data_from_numpy(image_np, binary_data=True)
        outputs = [httpclient.InferRequestedOutput(oname, binary_data=True) for oname in output_names]
        resp = client.infer(model_name=model_name, inputs=[inp], outputs=outputs)
        for oname in output_names:
            try:
                out_np = resp.as_numpy(oname)
                print(f"Output '{oname}' shape: {None if out_np is None else out_np.shape}")
            except Exception as e:
                print(f"Не удалось получить выход {oname}: {e}")
    except InferenceServerException as e:
        print("Ошибка Triton:", e)
    except Exception as e:
        print("Ошибка:", e)

def infer_empty(client: httpclient.InferenceServerClient, model_name: str, input_name: str, output_names: List[str]):
    """
    Отправляет "пустой" (нулевой) тензор с корректной формой.
    Triton/модель обычно ожидает батч-ось — поэтому используем (1,3,H,W) нули.
    """
    print("\n--- infer_empty ---")
    shape = (BATCH_SIZE, 3, TARGET_H, TARGET_W)
    zero_image = np.zeros(shape, dtype=np.float32)
    try:
        inp = httpclient.InferInput(input_name, zero_image.shape, "FP32")
        inp.set_data_from_numpy(zero_image, binary_data=True)
        outputs = [httpclient.InferRequestedOutput(oname, binary_data=True) for oname in output_names]
        resp = client.infer(model_name=model_name, inputs=[inp], outputs=outputs)
        for oname in output_names:
            try:
                out_np = resp.as_numpy(oname)
                print(f"Output '{oname}' shape: {None if out_np is None else out_np.shape}")
            except Exception as e:
                print(f"Не удалось получить выход {oname}: {e}")
    except InferenceServerException as e:
        print("Ошибка Triton:", e)
    except Exception as e:
        print("Ошибка:", e)

def main():
    # Инициализация клиента
    client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    # Получаем имена input/output из metadata Triton (если доступно)
    input_name, output_names = get_model_io(client, MODEL_NAME)
    print(f"Using input_name='{input_name}', output_names={output_names}")

    # 1) Отправляем мок-данные (рандом)
    infer_random(client, MODEL_NAME, input_name, output_names)

    # 2) Отправляем реальную фотографию.
    # Замените путь на существующий файл .jpg/.png в вашей системе для теста.
    sample_image_path = "img.png"
    infer_image(client, MODEL_NAME, input_name, output_names, sample_image_path)

    # 3) Отправляем пустые (нулевые) данные
    infer_empty(client, MODEL_NAME, input_name, output_names)

if __name__ == "__main__":
    main()
