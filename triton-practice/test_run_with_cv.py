#!/usr/bin/env python3
"""
test_run_with_cv.py

3 примера инференса в Triton + постобработка + отрисовка результатов через OpenCV.

Функции:
- infer_random_cv()
- infer_image_cv(image_path)
- infer_empty_cv()

Скрипт пытается автоматически извлечь имена входа/выходов из Triton (get_model_metadata).
Параметры: TRITON_URL, MODEL_NAME, TARGET_H/W, BATCH_SIZE.
"""
import os
import math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import cv2
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

# ------------------------
TRITON_URL = "localhost:8000"
MODEL_NAME = "yolo11s"

TARGET_H = 640
TARGET_W = 640
BATCH_SIZE = 1

SCORE_THRESH = 0.25
NMS_IOU_THRESH = 0.45
SAVE_OUTPUT_DIR = "out"
os.makedirs(SAVE_OUTPUT_DIR, exist_ok=True)

# Класс-лейблы (пример для COCO 80 classes). Если у вас другая сигнатура/классы - замените.
COCO80 = [
    "__background__", "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep",
    "cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
    "hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]


# ------------------------
def get_model_io(client: httpclient.InferenceServerClient, model_name: str) -> Tuple[str, List[str]]:
    """Попытка получить имена input/output из Triton metadata."""
    try:
        md = client.get_model_metadata(model_name=model_name)
        inputs = [inp.name for inp in md.inputs]
        outputs = [out.name for out in md.outputs]
        print(f"[meta] inputs: {inputs}, outputs: {outputs}")
        input_name = inputs[0] if inputs else "images"
        return input_name, outputs or ["output0"]
    except Exception as e:
        print(f"[meta] не удалось получить metadata: {e}. Использую дефолты 'images'/'output0'.")
        return "images", ["output0"]


def prepare_image_np(img: Image.Image, target_h: int, target_w: int) -> np.ndarray:
    """
    Простой ресайз + нормализация [0..1] и приведение к форме (B, C, H, W).
    (Вместо простого ресайза можно вставить letterbox, если нужно сохранять соотношение).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((target_w, target_h), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # H,W,C
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    arr = np.expand_dims(arr, axis=0)  # 1,C,H,W
    return arr


def xywhc_to_xyxy(box: Tuple[float, float, float, float], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Преобразует (cx,cy,w,h) в (x1,y1,x2,y2); осторожно с нормированными координатами."""
    cx, cy, w, h = box
    # Детектируем — нормированные ли координаты (<=1) или в пикселях
    if max(cx, cy, w, h) <= 1.01:
        # нормированные
        cx *= img_w
        w *= img_w
        cy *= img_h
        h *= img_h
    # иначе считаем, что уже в пикселях
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    # приводим к int и в пределах изображения
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img_w - 1, int(round(x2)))
    y2 = min(img_h - 1, int(round(y2)))
    return x1, y1, x2, y2


def postprocess_and_draw(orig_image: Image.Image, output_np: np.ndarray, out_name: str = "output0"):
    """
    Постпроцесс выходной тензорной матрицы и отрисовка.
    output_np expected shape: (B, C, N) or (C, N) or (B, C, N) with B==1.
    Возвращает изображение с отрисовкой (OpenCV BGR).
    """
    # Приводим форму к (B, C, N)
    if output_np is None:
        print("[post] output is None")
        return None
    arr = output_np
    if arr.ndim == 2:
        # (C, N) -> (1, C, N)
        arr = np.expand_dims(arr, axis=0)
    if arr.ndim != 3:
        raise ValueError(f"[post] неожиданный размер output: {arr.shape}")
    B, C, N = arr.shape
    print(f"[post] output shape (B,C,N): {arr.shape}")

    # Транспонируем для удобства -> detections: (B, N, C)
    detections = np.transpose(arr, (0, 2, 1))  # (B,N,C)
    img_w, img_h = orig_image.size
    img_cv = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)

    boxes_all = []
    scores_all = []
    classes_all = []

    for b in range(B):
        dets = detections[b]  # (N, C)
        # Простейшая эвристика для разбора формата:
        # - Если C >= 6, считаем, что первые 4: x,y,w,h. Оставшиеся либо [obj, class...] либо [class...]
        for i in range(dets.shape[0]):
            v = dets[i]
            # skip NaNs
            if np.isnan(v).any():
                continue
            # Основная эвристика:
            if C >= 6:
                # Попытка детектировать наличие objectness-channel (obj)
                # Если второй блок (индекс 4) типично в диапазоне [0..1], возможно это obj.
                obj_cand = v[4]
                # determine class scores start index
                if C >= 6 and (np.all((v[4:] >= 0) & (v[4:] <= 1)) and (C - 5) >= 1):
                    # возможно есть obj в v[4] и классы v[5:]
                    # but if number of class channels > 1: choose that split
                    # heuristic: if v[5:] has more than one element -> use obj*max(classes)
                    if v.shape[0] - 5 >= 1:
                        classes = v[5:]
                        obj = sigmoid(obj_cand) if (obj_cand < -0.0001 or obj_cand > 1.0001) else obj_cand
                        class_id = int(np.argmax(classes))
                        class_score = float(classes[class_id])
                        score = float(obj * class_score) if (0 <= obj <= 1) else float(class_score)
                    else:
                        # fallback: no class channels after obj -> treat v[4:] as classes
                        classes = v[4:]
                        class_id = int(np.argmax(classes))
                        class_score = float(classes[class_id])
                        score = class_score
                else:
                    # assume classes start at index 4
                    classes = v[4:]
                    class_id = int(np.argmax(classes)) if classes.size > 0 else -1
                    class_score = float(classes[class_id]) if class_id >= 0 else 0.0
                    score = class_score
                # bbox:
                cx, cy, w, h = float(v[0]), float(v[1]), float(v[2]), float(v[3])
            elif C == 5:
                # format maybe [x,y,w,h,conf] but no classes -> skip
                cx, cy, w, h = float(v[0]), float(v[1]), float(v[2]), float(v[3])
                class_id = 0
                score = float(v[4])
            else:
                # Unexpected channels count
                continue

            if score < SCORE_THRESH:
                continue

            x1, y1, x2, y2 = xywhc_to_xyxy((cx, cy, w, h), img_w, img_h)
            boxes_all.append([x1, y1, x2 - x1, y2 - y1])  # NMS expects [x,y,w,h]
            scores_all.append(score)
            classes_all.append(class_id if class_id >= 0 else 0)

    # Если нет детекций - возвращаем исходное изображение
    if not boxes_all:
        print("[post] нет детекций выше порога")
        return img_cv

    # NMS: используем cv2.dnn.NMSBoxes
    boxes_np = boxes_all
    scores_np = scores_all
    indices = cv2.dnn.NMSBoxes(boxes_np, scores_np, SCORE_THRESH, NMS_IOU_THRESH)
    picked = []
    if isinstance(indices, (list, tuple, np.ndarray)):
        # OpenCV иногда возвращает array of shape (k,1)
        try:
            indices_flat = indices.flatten().tolist()
        except:
            indices_flat = [int(x) for x in indices]
    else:
        indices_flat = []
    # If cv2 returned nested list, normalize:
    if len(indices) > 0 and hasattr(indices[0], '__len__'):
        indices_flat = [int(i[0]) for i in indices]

    if not indices_flat:
        # fallback: if cv2 didn't return, pick all
        indices_flat = list(range(len(boxes_np)))

    # рисуем
    for idx in indices_flat:
        x, y, w, h = boxes_np[idx]
        score = scores_np[idx]
        cls = classes_all[idx]
        label = COCO80[cls] if cls < len(COCO80) else f"class{cls}"
        # цвет по классу
        color = tuple(int(x) for x in np.random.RandomState(cls).randint(0, 255, size=3))
        # rectangle expects top-left and bottom-right
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 2)
        text = f"{label} {score:.2f}"
        # put text background
        ((tw, th), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_cv, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(img_cv, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_cv


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if abs(x) < 50 else (0.0 if x < 0 else 1.0)


# ------------------------
def infer_and_visualize(client: httpclient.InferenceServerClient,
                        model_name: str,
                        input_name: str,
                        output_names: List[str],
                        input_np: np.ndarray,
                        orig_image_for_draw: Image.Image,
                        save_name_suffix: str = "run"):
    """
    Универсальная: отправляет input_np в Triton и визуализирует первый output (output_names[0]).
    """
    try:
        inp = httpclient.InferInput(input_name, input_np.shape, "FP32")
        inp.set_data_from_numpy(input_np, binary_data=True)
        outputs = [httpclient.InferRequestedOutput(oname, binary_data=True) for oname in output_names]
        resp = client.infer(model_name=model_name, inputs=[inp], outputs=outputs)
        # Берём первый выход
        out_np = resp.as_numpy(output_names[0])
        print(f"[infer] recibed output '{output_names[0]}' shape: {None if out_np is None else out_np.shape}")
        img_cv = postprocess_and_draw(orig_image_for_draw, out_np, out_name=output_names[0])
        if img_cv is None:
            print("[infer] ничего рисовать")
            return
        out_file = os.path.join(SAVE_OUTPUT_DIR, f"{model_name}_{save_name_suffix}.jpg")
        cv2.imwrite(out_file, img_cv)
        print(f"[infer] сохранено: {out_file}")
        # Показываем через OpenCV окно
        window_name = f"{model_name}-{save_name_suffix}"
        cv2.imshow(window_name, img_cv)
        print("Press any key in the image window to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    except InferenceServerException as e:
        print("[infer] Triton error:", e)
    except Exception as e:
        print("[infer] Error:", e)


def infer_random_cv():
    print("\n=== infer_random_cv ===")
    client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    input_name, output_names = get_model_io(client, MODEL_NAME)
    shape = (BATCH_SIZE, 3, TARGET_H, TARGET_W)
    input_np = np.random.rand(*shape).astype(np.float32)
    # Для визуализации создаём пустое RGB изображение (для отрисовки результатов).
    orig_img = Image.fromarray((np.ones((TARGET_H, TARGET_W, 3)) * 255).astype(np.uint8))
    infer_and_visualize(client, MODEL_NAME, input_name, output_names, input_np, orig_img, save_name_suffix="random")


def infer_image_cv(image_path: str):
    print("\n=== infer_image_cv ===")
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    input_name, output_names = get_model_io(client, MODEL_NAME)
    img = Image.open(image_path).convert("RGB")
    input_np = prepare_image_np(img, TARGET_H, TARGET_W)
    infer_and_visualize(client, MODEL_NAME, input_name, output_names, input_np, img, save_name_suffix="image")


def infer_empty_cv():
    print("\n=== infer_empty_cv ===")
    client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    input_name, output_names = get_model_io(client, MODEL_NAME)
    shape = (BATCH_SIZE, 3, TARGET_H, TARGET_W)
    input_np = np.zeros(shape, dtype=np.float32)
    orig_img = Image.fromarray((np.ones((TARGET_H, TARGET_W, 3)) * 255).astype(np.uint8))
    infer_and_visualize(client, MODEL_NAME, input_name, output_names, input_np, orig_img, save_name_suffix="empty")


# ------------------------
def main():
    # пример: запускаем 3 варианта
    infer_random_cv()
    # поменяй путь на свой реальный файл:
    sample_image = "img.png"
    infer_image_cv(sample_image)
    infer_empty_cv()


if __name__ == "__main__":
    main()
