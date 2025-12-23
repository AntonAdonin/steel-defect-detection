"""Test MLflow API endpoints."""

import base64
import io
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image


def save_prediction(data, endpoint_name, image_path=None, img_bytes=None):
    """
    Сохраняет результат предсказания в JSON и изображение с bbox.

    Args:
        data: Словарь с результатом предсказания
        endpoint_name: Имя эндпоинта (для файла)
        image_path: Путь к исходной картинке (если есть)
        img_bytes: Альтернатива image_path для base64 (bytes)
    """
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Сохраняем JSON
    json_file = output_dir / f"{endpoint_name}_{timestamp}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f" Prediction JSON saved: {json_file}")

    # Загружаем изображение
    img = None
    if image_path and Path(image_path).exists():
        img = cv2.imread(str(image_path))
    elif img_bytes:
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    if img is not None:
        for det in data.get("detections", []):
            x1, y1, x2, y2 = det["bbox"]
            label = det.get("class_name", str(det.get("class", "")))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        img_file = output_dir / f"{endpoint_name}_img_{timestamp}.jpg"
        cv2.imwrite(str(img_file), img)
        print(f" Image with detections saved: {img_file}")


def test_api(api_url: str = "http://localhost:8000"):
    """
    Test MLflow API.

    Args:
        api_url: API base URL
    """
    print("=" * 80)
    print("Testing MLflow API")
    print("=" * 80)
    print(f"API URL: {api_url}\n")

    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"    Status: {data['status']}")
            print(f"    Model loaded: {data['model_loaded']}")
            print(f"    Model: {data['model_name']} ({data['model_stage']})")
        else:
            print(f"    Failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("    Cannot connect to API. Is it running?")
        print("      Start with: python commands.py start_api")
        return
    except Exception as e:
        print(f"    Error: {e}")
        return

    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        print(f"    Name: {data['name']}")
        print(f"    Version: {data['version']}")
        print(f"    Endpoints: {len(data['endpoints'])}")
    else:
        print(f"    Failed: {response.status_code}")

    # Test 3: Prediction with test image
    print("\n3. Testing prediction...")

    # Find a test image
    test_image_path = None
    for ext in ["jpg", "jpeg", "png"]:
        for path in Path("data").rglob(f"*.{ext}"):
            test_image_path = path
            break
        if test_image_path:
            break

    if not test_image_path or not test_image_path.exists():
        print("   ⚠️  No test image found. Creating dummy image...")
        # Create a dummy image
        import numpy as np

        dummy_img = np.random.randint(0, 255, (256, 1600, 3), dtype=np.uint8)
        img = Image.fromarray(dummy_img.astype("uint8"))
        test_image_path = Path("test_image.jpg")
        img.save(test_image_path)
        print(f"    Created: {test_image_path}")
    else:
        print(f"    Using image: {test_image_path}")

    # Test image upload
    print("   Testing /predict/image endpoint...")
    try:
        with open(test_image_path, "rb") as f:
            files = {"file": ("image.jpg", f, "image/jpeg")}
            response = requests.post(f"{api_url}/predict/image", files=files)

        if response.status_code == 200:
            data = response.json()
            print("    Success!")
            print(f"     - Detections: {data['num_detections']}")
            print(f"     - Processing time: {data['processing_time_ms']:.2f} ms")
            if data["detections"]:
                print("     - First detection:")
                det = data["detections"][0]
                print(f"       • BBox: {det['bbox']}")
                print(f"       • Confidence: {det['confidence']:.3f}")
                print(f"       • Class: {det['class_name']}")
            save_prediction(data, "predict_image", image_path=test_image_path)
        else:
            print(f"    Failed: {response.status_code}")
            print(f"      {response.text}")
    except Exception as e:
        print(f"    Error: {e}")

    # Test base64 endpoint
    print("\n   Testing /predict/base64 endpoint...")
    try:
        # Convert image to base64
        with open(test_image_path, "rb") as f:
            img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        response = requests.post(
            f"{api_url}/predict/base64",
            json={"image_base64": img_base64},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            data = response.json()
            print("    Success!")
            print(f"     - Detections: {data['num_detections']}")
            print(f"     - Processing time: {data['processing_time_ms']:.2f} ms")
            save_prediction(data, "predict_base64", img_bytes=img_bytes)
        else:
            print(f"    Failed: {response.status_code}")
            print(f"      {response.text}")
    except Exception as e:
        print(f"    Error: {e}")

    # Test 4: Model info
    print("\n4. Testing model info endpoint...")
    response = requests.get(f"{api_url}/model/info")
    if response.status_code == 200:
        data = response.json()
        print(f"    Model: {data['model_name']}")
        print(f"    Stage: {data['model_stage']}")
        print(f"    URI: {data['model_uri']}")
    else:
        print(f"    Failed: {response.status_code}")

    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)
    print(f"\nAPI Documentation: {api_url}/docs")
    print("=" * 80)


if __name__ == "__main__":
    import fire

    fire.Fire(test_api)
