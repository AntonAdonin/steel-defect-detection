from ultralytics.utils.benchmarks import benchmark
# https://docs.ultralytics.com/ru/modes/benchmark/#export-formats
benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, format="coreml")