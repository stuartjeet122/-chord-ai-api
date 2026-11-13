# Phase 6 AI infrastructure

## Status and boundaries

The API and worker infrastructure for model discovery, lifecycle management, isolated inference, normalized outputs, persistence, and chord/note benchmarking is implemented. **No production model provider is currently registered or enabled.** The catalog entries are research candidates, not promises of runnable models. Inference and benchmark submission therefore return `503 model_unavailable` until a provider is separately installed, configured, licensed, and validated. No model weights are downloaded automatically.

The model comparison and evidence limits are documented in [AI_MODEL_EVALUATION.md](./AI_MODEL_EVALUATION.md). In particular, repository code licenses do not establish pretrained-checkpoint rights, and the evaluation makes no unsupported accuracy or hardware-memory claims.

## Runtime and worker isolation

- The FastAPI process owns only the API-facing AI runtime, registry, and hardware/status reporting. It does not initialize a model during startup or execute model inference in a request.
- Inference and benchmark execution are queued as durable processing jobs. The outbox dispatcher routes `ai_inference` and `ai_benchmark` to the Celery `ai_inference` queue; audio ingestion remains on `audio_cpu`.
- The optional Compose AI worker uses concurrency one, a read-only model directory mount, CPU execution, and disabled downloads. This is a safe CPU-oriented scaffold, not a GPU-ready production image. A GPU deployment needs a separately validated image/runtime and device passthrough.
- Provider interfaces are task-specific and registered by provider factory. A provider must normalize outputs to the API's timestamped chord or note event schema before the inference engine accepts them.
- Model loading is lazy and worker-local. The lifecycle manager checks configured weight paths and optional SHA-256 values before loading. Model downloads are disabled by default; enabling downloads alone does not provide an approved downloader or authorize a checkpoint.
- Device selection supports `auto`, `cpu`, `cuda`, and `mps`. Explicit unavailable devices fail. CPU and accelerator inference have separate bounded semaphores; accelerator concurrency is one per worker process.
- Timeouts, cancellation checks, output validation, device/runtime metrics, and terminal job states are persisted. A process restart discards worker-local model state; durable job recovery is handled by the existing job/outbox recovery machinery.

Relevant settings:

| Setting | Default | Purpose |
|---|---:|---|
| `AI_MODEL_DIR` | `./models` | Local read-only model/checkpoint root used by workers. |
| `AI_EXECUTION_DEVICE` | `auto` | Preferred inference device (`auto`, `cpu`, `cuda`, or `mps`). |
| `AI_INFERENCE_MAX_CONCURRENCY` | `1` | Maximum concurrent CPU inference operations per worker. |
| `AI_INFERENCE_TIMEOUT_SECONDS` | `21600` | Per-inference timeout. |
| `AI_ALLOW_MODEL_DOWNLOADS` | `false` | Download gate; downloading remains unsupported until a verified provider-specific mechanism is added. |

## API

Model discovery and status do not load weights:

- `GET /api/v1/ai/models`
- `GET /api/v1/ai/models/{model_id}`
- `GET /api/v1/ai/models/{model_id}/status`
- `GET /api/v1/ai/hardware`

Inference submission uses an owned, available audio asset and returns a normal job response:

```json
{
  "asset_id": "UUID",
  "model_id": "registered-model-id",
  "task": "chord_recognition",
  "parameters": {}
}
```

Use `POST /api/v1/ai/inference` with an optional `Idempotency-Key`, then poll `GET /api/v1/ai/inference/{job_id}` for normalized results and execution metrics. The endpoint rejects unregistered models, unsupported model/task combinations, unavailable providers, non-audio assets, and unavailable assets before enqueueing work.

### Benchmark datasets

Datasets are caller-owned records whose samples reference audio assets already uploaded to the API. Registration accepts only available audio assets owned by the caller and stores event annotations, not duplicate audio bytes. Dataset IDs and versions are unique per owner.

```json
{
  "dataset_id": "my-chord-evaluation",
  "version": "1",
  "task": "chord_recognition",
  "license": "CC-BY-4.0",
  "attribution": "Required dataset attribution",
  "samples": [
    {
      "asset_id": "UUID",
      "references": [
        {
          "start_seconds": 0.0,
          "end_seconds": 1.8,
          "symbol": "C:maj",
          "root": "C",
          "quality": "major"
        }
      ]
    }
  ]
}
```

- `POST /api/v1/ai/benchmarks/datasets` registers a dataset.
- `GET /api/v1/ai/benchmarks/datasets?limit=50&offset=0` lists the caller's datasets.
- `POST /api/v1/ai/benchmarks` queues evaluation for an exact `dataset_id` and `dataset_version`, task, and one or more registered model IDs.
- `GET /api/v1/ai/benchmarks/{job_id}` returns run status and per-model sample/aggregate metrics.

Benchmark samples currently support chord events (`start_seconds`, `end_seconds`, `symbol`, `root`, `quality`, optional `bass`/`inversion`) and note events (`start_seconds`, `end_seconds`, `pitch_midi`). Submission is bounded to 256 unique audio assets per dataset and 20,000 reference events per sample. Chord metrics report duration-weighted symbol/root/quality accuracy and chord-change timing; note metrics report pitch/onset/offset precision and recall measures. Per-sample metrics are retained alongside aggregates; aggregates average rates/errors and sum event counts. These metrics are not a substitute for an explicitly defined, representative product benchmark.

The benchmark dataset's stated license and attribution are recorded as metadata supplied by the caller; the API does not verify rights to the dataset or recordings. Register only data you are authorized to process.

## Persistence and migration

Inference records, benchmark dataset metadata/annotations, and benchmark runs are stored in the `ai_*` tables introduced by the Phase 6 Alembic migration. Job status remains in the shared processing-job tables. Dataset ownership is enforced on list, lookup, and audio-asset validation. Deleting source audio referenced by benchmark records is constrained by database foreign keys.

## Not yet implemented or validated

- No model adapter is registered; there is no verified end-to-end inference against third-party weights.
- Batch inference and inference-result cache execution are not exposed; their persistence groundwork must not be interpreted as active caching.
- Benchmarking is currently limited to chord recognition and note transcription; source-separation metric utilities are not wired into benchmark dataset runs.
- The candidate model runtime, Python 3.12+ installation compatibility, exact checkpoint commercial terms, GPU image/device setup, model accuracy, latency, and peak memory remain unverified.
- Authentication is not yet a production identity system; ownership currently follows the application's existing development identity mechanism.
- WebSocket progress is not part of this phase; clients poll the existing job/result endpoints.

Do not enable a model in the catalog or accept its outputs as production analysis until checkpoint rights, runtime compatibility, representative benchmark results, and operational resource limits are verified and documented.
