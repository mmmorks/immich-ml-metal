"""Lock that serializes MLX (CLIP) inference and model unloads.

MLX's Metal backend is built for single-threaded-per-stream submission:
command encoders are thread-local and each stream's GPU work is driven by
one dedicated worker thread. We call CLIP from a multi-threaded request
pool on the shared default GPU stream, so two things must be serialized or
the Metal command-buffer lifecycle is raced and the process aborts with
``'addCompletedHandler after commit'``:

  1. Concurrent CLIP evaluations on that stream.
  2. A CLIP eval racing a model swap — ``unload()`` calls ``mx.clear_cache()``,
     returning the buffer pool to the allocator; doing that while a lazy MLX
     array is still materializing is the same collision.

So CLIP forces materialization (``np.array``) *inside* this lock, and
``unload()`` takes the same lock.

This is NOT about MLX racing Vision: the lock is acquired only by CLIP.
Vision framework (face detection, OCR) and CoreML (face embeddings via
ONNX+CoreML EP) are independently thread-safe with their own Metal command
queues, so they run lock-free and overlap CLIP freely — that cross-engine
parallelism is exactly what we want.

Import: from src.gpu_lock import metal_lock
"""

import threading

metal_lock = threading.Lock()
