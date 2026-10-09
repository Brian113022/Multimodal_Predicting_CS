"""Run only prepared feature jobs concurrently and merge their progress."""

from concurrent.futures import ThreadPoolExecutor
import logging
from queue import Queue
import threading

from .errors import ExtractionError
from .worker import terminate_worker


def parallel_feature_events(factories):
    """Yield stage/result/error events for up to two independent feature jobs.

    Factories accept on_process and must return extraction-only generators.
    Closing this generator stops registered subprocess trees before callers
    can delete the temporary images and masks they are reading.
    """
    if not factories:
        return
    messages = Queue()
    cancelled = threading.Event()
    process_lock = threading.Lock()
    processes = []

    def register(process):
        with process_lock:
            if cancelled.is_set():
                terminate_worker(process)
            else:
                processes.append(process)

    def run(modality, factory):
        events = None
        try:
            events = factory(on_process=register)
            while not cancelled.is_set():
                try:
                    stage = next(events)
                except StopIteration as stop:
                    messages.put({"event": "result", "modality": modality, "features": stop.value})
                    return
                messages.put({**stage, "event": "stage", "phase": "feature_extraction", "modality": modality})
        except Exception as exc:
            if isinstance(exc, ExtractionError):
                message = str(exc)
            else:
                logging.getLogger(__name__).exception("%s feature extraction failed", modality)
                message = f"Unexpected {modality} feature extraction failure."
            messages.put({"event": "modality_error", "modality": modality, "message": message})
        finally:
            if events is not None:
                events.close()

    pool = ThreadPoolExecutor(max_workers=min(2, len(factories)), thread_name_prefix="mri-features")
    try:
        for modality, factory in factories.items():
            pool.submit(run, modality, factory)
        finished = 0
        while finished < len(factories):
            event = messages.get()
            if event["event"] in {"result", "modality_error"}:
                finished += 1
            yield event
    finally:
        cancelled.set()
        with process_lock:
            for process in processes:
                terminate_worker(process)
        pool.shutdown(wait=True, cancel_futures=True)
