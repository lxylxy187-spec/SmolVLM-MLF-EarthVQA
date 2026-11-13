from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

_THREAD_LOCAL = threading.local()


def build_worker_processors(model_dir: Path, count: int) -> list[Any]:
    """Load an independent deterministic processor per encode thread."""
    if count <= 0:
        return []
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    try:
        return [
            AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
            for _ in range(count)
        ]
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def _initialize_encode_thread(processors: queue.SimpleQueue, encode_fn: Callable) -> None:
    _THREAD_LOCAL.processor = processors.get()
    _THREAD_LOCAL.encode_fn = encode_fn


def _encode_in_thread(example: dict[str, Any]):
    processor = getattr(_THREAD_LOCAL, "processor", None)
    encode_fn = getattr(_THREAD_LOCAL, "encode_fn", None)
    if processor is None or encode_fn is None:
        raise RuntimeError("Ordered preprocessing worker was not initialized")
    return encode_fn(processor, example)


def ordered_encoded_batches(
    loader: DataLoader,
    skip_batches: int,
    serial_processor: Any,
    worker_processors: list[Any],
    max_inflight: int,
    encode_fn: Callable,
):
    """Pre-encode on CPU while yielding examples in sampler order."""
    source = iter(enumerate(loader))
    for _ in range(skip_batches):
        try:
            next(source)
        except StopIteration as error:
            raise RuntimeError("Resume skip exceeds the epoch DataLoader") from error

    if not worker_processors:
        for batch_index, example in source:
            started = time.perf_counter()
            encoded = encode_fn(serial_processor, example)
            yield batch_index, example, encoded, time.perf_counter() - started
        return

    processor_queue: queue.SimpleQueue = queue.SimpleQueue()
    for processor in worker_processors:
        processor_queue.put(processor)
    executor = ThreadPoolExecutor(
        max_workers=len(worker_processors),
        initializer=_initialize_encode_thread,
        initargs=(processor_queue, encode_fn),
        thread_name_prefix="earthvqa-encode",
    )
    pending: deque[tuple[int, dict[str, Any], Any]] = deque()

    def fill() -> None:
        while len(pending) < max_inflight:
            try:
                batch_index, example = next(source)
            except StopIteration:
                break
            pending.append(
                (batch_index, example, executor.submit(_encode_in_thread, example))
            )

    try:
        fill()
        while pending:
            batch_index, example, future = pending.popleft()
            wait_started = time.perf_counter()
            try:
                encoded = future.result()
            except Exception as error:
                raise RuntimeError(
                    "Ordered CPU preprocessing failed for "
                    f"batch={batch_index}, image={example.get('image_id')}, "
                    f"qa_index={example.get('qa_index')}"
                ) from error
            data_wait = time.perf_counter() - wait_started
            yield batch_index, example, encoded, data_wait
            fill()
    finally:
        for _, _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def compute_sample_order_sha256(dataset: Any, seed: int, collate_fn: Callable) -> tuple[str, int]:
    """Hash the exact ordered (image_id, qa_index) sequence used by training."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_fn,
    )
    digest = hashlib.sha256()
    count = 0
    for example in loader:
        digest.update(
            f"{example['image_id']}\t{int(example['qa_index'])}\n".encode("utf-8")
        )
        count += 1
    return digest.hexdigest(), count
