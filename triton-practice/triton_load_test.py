#!/usr/bin/env python3
"""
triton_load_test.py

Простой нагрузочный тест для Triton Inference Server (HTTP client).

Пример:
python triton_load_test.py --url localhost:8000 --model yolo11s --clients 8 --requests 100 --batch 1 --height 640 --width 640

Опции:
  --clients         : количество параллельных клиентов (потоков)
  --requests        : запросов на клиент (каждый клиент пошлёт это количество запросов)
  --batch           : batch size в запросе (B)
  --height, --width : H и W входного тензора (C фиксированно = 3)
  --zero            : если указан, отправляет нулевые тензоры вместо рандома
  --input-name      : имя входного тензора (по умолчанию "images")
  --output-name     : имя выходного тензора (по умолчанию "output0")
  --verbose         : печатать лог каждой ошибки
"""
import argparse
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

# -----------------------
def worker_loop(worker_id: int,
                url: str,
                model: str,
                input_name: str,
                output_name: str,
                batch: int,
                height: int,
                width: int,
                requests_per_worker: int,
                use_zero: bool,
                verbose: bool,
                results_list: List[float],
                errors_counter: List[int],
                lock: threading.Lock):
    """
    Один worker: создаёт своего клиента и последовательно шлёт requests_per_worker запросов.
    Сохраняет в results_list латентности (ms) для успешных запросов.
    errors_counter — список с одним элементом, инкрементируемым при ошибке (мутекс нужен).
    """
    try:
        client = httpclient.InferenceServerClient(url=url, verbose=False)
    except Exception as e:
        with lock:
            errors_counter[0] += requests_per_worker
        if verbose:
            print(f"[W{worker_id}] failed to create client: {e}")
        return

    # предсоздаём входной тензор (повторно используем для всех запросов, изменяя только содержимое)
    shape = (batch, 3, height, width)
    # prepare requested output handle
    req_output = httpclient.InferRequestedOutput(output_name, binary_data=True)

    for i in range(requests_per_worker):
        # генерируем данные
        if use_zero:
            arr = np.zeros(shape, dtype=np.float32)
        else:
            # random in [0,1)
            arr = np.random.rand(*shape).astype(np.float32)

        try:
            infer_input = httpclient.InferInput(input_name, arr.shape, "FP32")
            infer_input.set_data_from_numpy(arr, binary_data=True)

            t0 = time.perf_counter()
            resp = client.infer(model_name=model,
                                inputs=[infer_input],
                                outputs=[req_output])
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            # try to validate response presence quickly (may be None for some models/backends)
            try:
                _ = resp.as_numpy(output_name)
            except Exception:
                # если не можем получить выход — считаем как ошибка, но всё равно замерили latency
                with lock:
                    errors_counter[0] += 1
                if verbose:
                    print(f"[W{worker_id}] resp.as_numpy('{output_name}') failed on req {i}")
                continue

            with lock:
                results_list.append(latency_ms)

        except InferenceServerException as e:
            with lock:
                errors_counter[0] += 1
            if verbose:
                print(f"[W{worker_id}] Triton error on req {i}: {e}")
        except Exception as e:
            with lock:
                errors_counter[0] += 1
            if verbose:
                print(f"[W{worker_id}] Unexpected error on req {i}: {e}")

# -----------------------
def print_summary(latencies_ms: List[float], total_sent: int, errors: int, elapsed_s: float):
    succeeded = len(latencies_ms)
    print("\n=== Summary ===")
    print(f"Total requests attempted : {total_sent}")
    print(f"Successful responses     : {succeeded}")
    print(f"Failed responses         : {errors}")
    print(f"Total elapsed (wall)    : {elapsed_s:.3f} s")
    tps = succeeded / elapsed_s if elapsed_s > 0 else 0.0
    print(f"Throughput (successful req / sec): {tps:.2f}")

    if succeeded > 0:
        lat_sorted = sorted(latencies_ms)
        print(f"Latency ms - mean : {statistics.mean(lat_sorted):.2f}")
        print(f"Latency ms - median (p50): {statistics.median(lat_sorted):.2f}")
        def pct(p):
            idx = int(len(lat_sorted) * p / 100)
            idx = min(max(idx, 0), len(lat_sorted)-1)
            return lat_sorted[idx]
        print(f"Latency ms - p90 : {pct(90):.2f}")
        print(f"Latency ms - p95 : {pct(95):.2f}")
        print(f"Latency ms - p99 : {pct(99):.2f}")
        print(f"Min latency ms: {lat_sorted[0]:.2f}, Max latency ms: {lat_sorted[-1]:.2f}")
    else:
        print("No successful latencies recorded.")

# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Simple Triton load tester with parallel clients")
    parser.add_argument("--url", type=str, default="localhost:8000", help="Triton HTTP URL (host:port)")
    parser.add_argument("--model", type=str, default="yolo11s", help="Model name in Triton")
    parser.add_argument("--clients", type=int, default=4, help="Number of parallel clients (threads)")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests per client")
    parser.add_argument("--batch", type=int, default=1, help="Batch size per request (B)")
    parser.add_argument("--height", type=int, default=640, help="Input height (H)")
    parser.add_argument("--width", type=int, default=640, help="Input width (W)")
    parser.add_argument("--zero", action="store_true", help="Send zero tensors instead of random")
    parser.add_argument("--input-name", type=str, default="images", help="Input tensor name")
    parser.add_argument("--output-name", type=str, default="output0", help="Output tensor name")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-error logging")
    args = parser.parse_args()

    total_requests = args.clients * args.requests

    print("Triton load test config:")
    print(f" URL         : {args.url}")
    print(f" Model       : {args.model}")
    print(f" Clients     : {args.clients}")
    print(f" Requests/Cl : {args.requests}")
    print(f" Total reqs  : {total_requests}")
    print(f" Batch (B,C,H,W): ({args.batch},3,{args.height},{args.width})")
    print(f" Zero data   : {args.zero}")
    print(f" Input name  : {args.input_name}")
    print(f" Output name : {args.output_name}")
    print(" Starting...")

    # Result collectors (shared)
    latencies_ms: List[float] = []
    errors = [0]  # use list for mutability in closures
    lock = threading.Lock()

    start_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.clients) as ex:
        futures = []
        for i in range(args.clients):
            fut = ex.submit(worker_loop,
                             i,
                             args.url,
                             args.model,
                             args.input_name,
                             args.output_name,
                             args.batch,
                             args.height,
                             args.width,
                             args.requests,
                             args.zero,
                             args.verbose,
                             latencies_ms,
                             errors,
                             lock)
            futures.append(fut)

        # wait for all
        for f in as_completed(futures):
            try:
                _ = f.result()
            except Exception as e:
                print("Worker raised exception:", e)

    end_all = time.perf_counter()
    elapsed = end_all - start_all

    print_summary(latencies_ms, total_requests, errors[0], elapsed)

if __name__ == "__main__":
    main()
