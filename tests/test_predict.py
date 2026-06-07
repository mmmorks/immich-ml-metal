"""Tests for the /predict endpoint — concurrent task execution, response format."""

import asyncio
import json

import httpx
import pytest
from fastapi.responses import JSONResponse

import src.main as main
from src.main import app


@pytest.fixture
def client():
    """Async test client against the FastAPI app (no real server needed)."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _entries(*task_types):
    """Build an entries JSON for the given task types."""
    entries = {}
    if "clip" in task_types:
        entries["clip"] = {"visual": {"modelName": "ViT-B-32__openai"}}
    if "clip-text" in task_types:
        entries["clip"] = {"textual": {"modelName": "ViT-B-32__openai"}}
    if "facial-recognition" in task_types:
        entries["facial-recognition"] = {"detection": {}, "recognition": {}}
    if "ocr" in task_types:
        entries["ocr"] = {"detection": {}, "recognition": {}}
    return json.dumps(entries)


# --- Basic endpoint tests ---


@pytest.mark.asyncio
async def test_ping(client):
    resp = await client.get("/ping")
    assert resp.status_code == 200
    assert resp.text == "pong"


@pytest.mark.asyncio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"] == "Immich ML"


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["stub_mode"] is True


SECRET = "/Users/secret/path/model.bin not found"


@pytest.mark.asyncio
async def test_health_hides_error_details_without_debug(client, monkeypatch):
    """/health must not leak raw exception strings when debug_mode is off."""
    import src.main as main

    monkeypatch.setattr(main, "STUB_MODE", False)
    monkeypatch.setattr(main.settings, "debug_mode", False)

    def _boom(*args, **kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(main, "get_clip", _boom)

    resp = await client.get("/health")
    data = resp.json()
    # Degraded, but the raw exception string must not appear anywhere in the body.
    assert data["checks"]["clip"] == "error"
    assert SECRET not in json.dumps(data)


@pytest.mark.asyncio
async def test_health_reuses_loaded_clip_model(client, monkeypatch):
    """/health must probe the already-loaded CLIP model, not switch to
    settings.clip_model — otherwise it evicts the production model (e.g. SigLIP2)
    from the single CLIP slot on every probe and thrashes the cache."""
    import src.main as main
    import src.models.clip as clip_module
    import src.models.face_embed as face_module

    monkeypatch.setattr(main, "STUB_MODE", False)
    # Pretend SigLIP2 is the live production model in the single CLIP slot.
    LIVE = "ViT-SO400M-16-SigLIP2-384__webli"
    monkeypatch.setattr(clip_module, "_current_model_name", LIVE)
    monkeypatch.setattr(main.settings, "clip_model", "ViT-B-32__openai")

    # Record what model name the CLIP check actually requests.
    called = {}

    def _record_get_clip(model_name="ViT-B-32__openai"):
        called["name"] = model_name
        return object()

    monkeypatch.setattr(main, "get_clip", _record_get_clip)
    # Neutralize the unrelated sub-checks so they don't load real models.
    monkeypatch.setattr(face_module, "get_recognition_model", lambda *a, **k: object())

    resp = await client.get("/health")
    assert resp.status_code == 200
    assert called["name"] == LIVE, "health probed settings.clip_model instead of the loaded model — this evicts the production model"


@pytest.mark.asyncio
async def test_health_falls_back_to_settings_when_nothing_loaded(client, monkeypatch):
    """With no CLIP model loaded yet, /health probes settings.clip_model."""
    import src.main as main
    import src.models.clip as clip_module
    import src.models.face_embed as face_module

    monkeypatch.setattr(main, "STUB_MODE", False)
    monkeypatch.setattr(clip_module, "_current_model_name", None)
    monkeypatch.setattr(main.settings, "clip_model", "ViT-B-32__openai")

    called = {}

    def _record_get_clip(model_name="ViT-B-32__openai"):
        called["name"] = model_name
        return object()

    monkeypatch.setattr(main, "get_clip", _record_get_clip)
    monkeypatch.setattr(face_module, "get_recognition_model", lambda *a, **k: object())

    await client.get("/health")
    assert called["name"] == "ViT-B-32__openai"


@pytest.mark.asyncio
async def test_health_exposes_error_details_with_debug(client, monkeypatch):
    """With debug_mode on, the raw exception string is allowed through for diagnostics."""
    import src.main as main

    monkeypatch.setattr(main, "STUB_MODE", False)
    monkeypatch.setattr(main.settings, "debug_mode", True)

    def _boom(*args, **kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(main, "get_clip", _boom)

    resp = await client.get("/health")
    data = resp.json()
    assert data["checks"]["clip"] == f"error: {SECRET}"


# --- Predict: single tasks ---


@pytest.mark.asyncio
async def test_predict_clip_visual(client, test_image_bytes):
    resp = await client.post(
        "/predict",
        data={"entries": _entries("clip")},
        files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "clip" in data
    embedding = json.loads(data["clip"])
    assert len(embedding) == 512
    assert "imageHeight" in data
    assert "imageWidth" in data


@pytest.mark.asyncio
async def test_predict_clip_text(client):
    resp = await client.post(
        "/predict",
        data={"entries": _entries("clip-text"), "text": "a photo of a cat"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "clip" in data
    embedding = json.loads(data["clip"])
    assert len(embedding) == 512


@pytest.mark.asyncio
async def test_predict_faces(client, test_image_bytes):
    resp = await client.post(
        "/predict",
        data={"entries": _entries("facial-recognition")},
        files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "facial-recognition" in data
    faces = data["facial-recognition"]
    assert isinstance(faces, list)
    # Stub mode returns 1 fake face
    assert len(faces) == 1
    assert "boundingBox" in faces[0]
    assert "embedding" in faces[0]
    assert "score" in faces[0]
    # The face embedding must be a JSON string (same contract as the CLIP path),
    # not a Python repr — str(list) emits non-JSON 'nan'/'inf' tokens.
    face_embedding = json.loads(faces[0]["embedding"])
    assert isinstance(face_embedding, list)
    assert len(face_embedding) == 512


def test_serialize_embedding_produces_json():
    import numpy as np

    from src.main import _serialize_embedding

    # numpy array (CLIP/face inference path) and plain list (stub path) must
    # both round-trip through json.loads to the same values.
    arr = np.array([1.0, -0.5, 0.25], dtype=np.float32)
    assert json.loads(_serialize_embedding(arr)) == [1.0, -0.5, 0.25]
    assert json.loads(_serialize_embedding([1.0, 2.0, 3.0])) == [1.0, 2.0, 3.0]

    # A degenerate vector stays valid JSON (json.dumps emits NaN/Infinity,
    # which json.loads parses) rather than str()'s un-parseable 'nan'/'inf'.
    degenerate = json.loads(_serialize_embedding([float("nan"), float("inf")]))
    assert len(degenerate) == 2 and degenerate[0] != degenerate[0]  # NaN != NaN


@pytest.mark.asyncio
async def test_predict_ocr(client, test_image_bytes):
    resp = await client.post(
        "/predict",
        data={"entries": _entries("ocr")},
        files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "ocr" in data
    ocr = data["ocr"]
    assert "text" in ocr
    assert "box" in ocr
    assert "boxScore" in ocr
    assert "textScore" in ocr


# --- Predict: concurrent tasks ---


@pytest.mark.asyncio
async def test_predict_all_three_tasks(client, test_image_bytes):
    """All 3 tasks in one request — tests asyncio.gather path."""
    entries = json.dumps(
        {
            "clip": {"visual": {"modelName": "ViT-B-32__openai"}},
            "facial-recognition": {"detection": {}, "recognition": {}},
            "ocr": {"detection": {}, "recognition": {}},
        }
    )
    resp = await client.post(
        "/predict",
        data={"entries": entries},
        files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "clip" in data
    assert "facial-recognition" in data
    assert "ocr" in data
    assert "imageHeight" in data
    assert "imageWidth" in data


# --- Error handling ---


@pytest.mark.asyncio
async def test_predict_no_image_or_text(client):
    resp = await client.post("/predict", data={"entries": _entries("clip")})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_predict_invalid_json(client, test_image_bytes):
    resp = await client.post(
        "/predict",
        data={"entries": "not json"},
        files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_predict_empty_tasks(client, test_image_bytes):
    """Empty entries dict — should return 200 with just image dimensions."""
    resp = await client.post(
        "/predict",
        data={"entries": "{}"},
        files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "imageHeight" in data
    assert "clip" not in data


# --- Backpressure & timeout ---


@pytest.fixture
def reset_semaphore():
    """Isolate semaphore mutations so a test's custom sizing doesn't leak."""
    saved = main._request_semaphore
    main._request_semaphore = None
    try:
        yield
    finally:
        main._request_semaphore = saved


@pytest.mark.asyncio
async def test_slow_processing_is_not_cancelled_by_timeout(client, monkeypatch, reset_semaphore):
    """Once a slot is acquired, processing must run to completion even if it
    exceeds request_timeout. The timeout only bounds queue wait — wrapping the
    uncancellable thread-pool work in it would orphan a pool thread.
    """
    monkeypatch.setattr(main.settings, "request_timeout", 0.3)

    async def slow_process(entries, image, text):
        # Far longer than request_timeout, but the semaphore was free so no
        # queue wait occurred — this should NOT be cancelled.
        await asyncio.sleep(1.0)
        return JSONResponse({"slow": "done"})

    monkeypatch.setattr(main, "_process_predict", slow_process)

    resp = await client.post("/predict", data={"entries": _entries("clip")})
    assert resp.status_code == 200
    assert resp.json() == {"slow": "done"}


@pytest.mark.asyncio
async def test_queue_wait_times_out_with_503(client, monkeypatch, reset_semaphore):
    """When all slots are busy, a request that waits longer than request_timeout
    for a slot gets 503 — backpressure still works.
    """
    monkeypatch.setattr(main.settings, "max_concurrent_requests", 1)
    monkeypatch.setattr(main.settings, "request_timeout", 0.3)

    release = asyncio.Event()

    async def blocking_process(entries, image, text):
        await release.wait()
        return JSONResponse({"ok": True})

    monkeypatch.setattr(main, "_process_predict", blocking_process)

    # First request grabs the only slot and parks inside processing.
    holder = asyncio.create_task(client.post("/predict", data={"entries": _entries("clip")}))
    await asyncio.sleep(0.05)  # let the holder acquire the slot

    # Second request must wait for the slot and time out → 503.
    resp = await client.post("/predict", data={"entries": _entries("clip")})
    assert resp.status_code == 503
    assert "overloaded" in resp.json()["detail"].lower()

    # Let the holder finish cleanly.
    release.set()
    holder_resp = await holder
    assert holder_resp.status_code == 200


@pytest.mark.asyncio
async def test_semaphore_not_leaked_on_queue_timeout(client, monkeypatch, reset_semaphore):
    """After a queue-wait timeout, the slot must be reusable — no permit leak."""
    monkeypatch.setattr(main.settings, "max_concurrent_requests", 1)
    monkeypatch.setattr(main.settings, "request_timeout", 0.3)

    release = asyncio.Event()

    async def blocking_process(entries, image, text):
        await release.wait()
        return JSONResponse({"ok": True})

    monkeypatch.setattr(main, "_process_predict", blocking_process)

    holder = asyncio.create_task(client.post("/predict", data={"entries": _entries("clip")}))
    await asyncio.sleep(0.05)

    # This one times out waiting for the slot.
    timed_out = await client.post("/predict", data={"entries": _entries("clip")})
    assert timed_out.status_code == 503

    # Release the holder; the slot must be fully available again.
    release.set()
    assert (await holder).status_code == 200

    # A fresh request now sails through — proving the permit wasn't leaked.
    monkeypatch.setattr(main, "_process_predict", _passthrough_process)
    resp = await client.post("/predict", data={"entries": _entries("clip")})
    assert resp.status_code == 200


async def _passthrough_process(entries, image, text):
    return JSONResponse({"ok": True})
