from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from music_api.ai.models import (
    AIInferenceRecord,
    BenchmarkDatasetAssetRecord,
    BenchmarkDatasetRecord,
    BenchmarkRunRecord,
)
from music_api.ai.registry import ModelNotFoundError
from music_api.ai.schemas import (
    BenchmarkDatasetRequest,
    BenchmarkDatasetResponse,
    BenchmarkRequest,
    BenchmarkRunResponse,
    HardwareResponse,
    InferenceRecordResponse,
    InferenceRequest,
    ModelResponse,
    ModelStatusResponse,
)
from music_api.api.dependencies import (
    get_db_session,
    get_owner_id,
    get_resources,
    get_storage_service,
)
from music_api.audio.schemas import JobResponse
from music_api.errors import ApplicationError
from music_api.jobs.models import ProcessingJob
from music_api.jobs.service import JobConflictError, create_job
from music_api.resources import RuntimeResources
from music_api.storage.models import AssetStatus, AssetType, AudioAsset
from music_api.storage.service import AssetNotFoundError, StorageService

router = APIRouter(prefix="/ai", tags=["AI"])
Resources = Annotated[RuntimeResources, Depends(get_resources)]
OwnerId = Annotated[uuid.UUID, Depends(get_owner_id)]
DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]
Storage = Annotated[StorageService, Depends(get_storage_service)]


@router.get("/models", response_model=list[ModelResponse], summary="List registered AI models")
async def list_models(resources: Resources) -> list[ModelResponse]:
    return [ModelResponse.model_validate(definition) for definition in resources.ai.registry.list()]


@router.get(
    "/models/{model_id}",
    response_model=ModelResponse,
    summary="Get model metadata and compatibility information",
)
async def get_model(model_id: str, resources: Resources) -> ModelResponse:
    try:
        definition = resources.ai.registry.get(model_id)
    except ModelNotFoundError as exc:
        raise ApplicationError("model_not_found", str(exc), 404) from exc
    return ModelResponse.model_validate(definition)


@router.get(
    "/models/{model_id}/status",
    response_model=ModelStatusResponse,
    summary="Get model lifecycle and readiness status",
)
async def get_model_status(model_id: str, resources: Resources) -> ModelStatusResponse:
    try:
        definition = resources.ai.registry.get(model_id)
    except ModelNotFoundError as exc:
        raise ApplicationError("model_not_found", str(exc), 404) from exc
    return resources.ai.lifecycle.status(definition)


@router.get(
    "/hardware",
    response_model=HardwareResponse,
    summary="Get supported local inference hardware and runtimes",
)
async def get_hardware(resources: Resources) -> HardwareResponse:
    return HardwareResponse.model_validate(
        resources.ai.hardware.as_response(resources.settings.ai_execution_device.value)
    )


@router.post(
    "/inference",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue one model inference job",
)
async def queue_inference(
    request: InferenceRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    try:
        definition = resources.ai.registry.get(request.model_id)
    except ModelNotFoundError as exc:
        raise ApplicationError("model_not_found", str(exc), 404) from exc
    if request.task not in definition.supported_tasks:
        raise ApplicationError(
            "unsupported_model_task",
            f"Model '{definition.model_id}' does not support task '{request.task.value}'.",
            422,
        )
    if not resources.ai.provider_is_configured(definition.model_id):
        state = resources.ai.lifecycle.status(definition).state.value
        raise ApplicationError(
            "model_unavailable",
            f"Model '{definition.model_id}' is not enabled and ready for inference ({state}).",
            503,
        )
    try:
        asset = await storage.get_asset(session, owner_id, request.asset_id)
    except AssetNotFoundError as exc:
        raise ApplicationError("asset_not_found", str(exc), 404) from exc
    audio_types = {
        AssetType.ORIGINAL_AUDIO.value,
        AssetType.CONVERTED_AUDIO.value,
        AssetType.NORMALIZED_AUDIO.value,
        AssetType.TRIMMED_AUDIO.value,
        AssetType.PROCESSED_AUDIO.value,
        AssetType.GENERATED_AUDIO.value,
        AssetType.SEPARATED_STEM.value,
    }
    if asset.asset_type not in audio_types:
        raise ApplicationError("not_an_audio_asset", "The requested asset is not audio.", 415)
    if asset.status != AssetStatus.AVAILABLE.value:
        raise ApplicationError("asset_not_available", "The audio asset is not available.", 409)
    try:
        job, created = await create_job(
            session,
            owner_id,
            asset.id,
            "ai_inference",
            {
                "model_id": request.model_id,
                "task": request.task.value,
                "parameters": request.parameters,
            },
            idempotency_key,
        )
    except JobConflictError as exc:
        raise ApplicationError("job_conflict", str(exc), 409) from exc
    except ValueError as exc:
        raise ApplicationError("invalid_idempotency_key", str(exc), 422) from exc
    if created:
        response.headers["Location"] = f"/api/v1/jobs/{job.id}"
    return JobResponse.model_validate(job)


@router.get(
    "/inference/{job_id}",
    response_model=InferenceRecordResponse,
    summary="Get owned inference output and execution metrics",
)
async def get_inference_result(
    job_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> InferenceRecordResponse:
    job = await session.scalar(
        select(ProcessingJob).where(
            ProcessingJob.id == job_id,
            ProcessingJob.owner_id == owner_id,
            ProcessingJob.task_type == "ai_inference",
        )
    )
    if job is None:
        raise ApplicationError("inference_not_found", "Inference job was not found.", 404)
    record = await session.scalar(
        select(AIInferenceRecord).where(AIInferenceRecord.job_id == job_id)
    )
    if record is None:
        raise ApplicationError(
            "inference_result_pending",
            "The inference worker has not produced a result record yet.",
            409,
        )
    return InferenceRecordResponse.model_validate(record)


@router.post(
    "/benchmarks/datasets",
    response_model=BenchmarkDatasetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register an owned, annotated audio benchmark dataset",
)
async def register_benchmark_dataset(
    request: BenchmarkDatasetRequest,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> BenchmarkDatasetResponse:
    asset_ids = [sample.asset_id for sample in request.samples]
    assets = list(
        (
            await session.scalars(
                select(AudioAsset).where(
                    AudioAsset.owner_id == owner_id,
                    AudioAsset.id.in_(asset_ids),
                    AudioAsset.status == AssetStatus.AVAILABLE.value,
                    AudioAsset.asset_type.in_(
                        {
                            AssetType.ORIGINAL_AUDIO.value,
                            AssetType.CONVERTED_AUDIO.value,
                            AssetType.NORMALIZED_AUDIO.value,
                            AssetType.TRIMMED_AUDIO.value,
                            AssetType.PROCESSED_AUDIO.value,
                            AssetType.GENERATED_AUDIO.value,
                            AssetType.SEPARATED_STEM.value,
                        }
                    ),
                )
            )
        ).all()
    )
    assets_by_id = {asset.id: asset for asset in assets}
    if set(assets_by_id) != set(asset_ids):
        raise ApplicationError(
            "benchmark_assets_unavailable",
            "Every benchmark sample must reference an available audio asset owned by the caller.",
            422,
        )
    for sample in request.samples:
        asset_duration = assets_by_id[sample.asset_id].duration_seconds
        if asset_duration is None or any(
            float(reference["end_seconds"]) > asset_duration for reference in sample.references
        ):
            raise ApplicationError(
                "benchmark_annotation_out_of_range",
                "Every benchmark annotation must fit within a sample's known audio duration.",
                422,
            )
    existing = await session.scalar(
        select(BenchmarkDatasetRecord.id).where(
            BenchmarkDatasetRecord.owner_id == owner_id,
            BenchmarkDatasetRecord.dataset_id == request.dataset_id,
            BenchmarkDatasetRecord.version == request.version,
        )
    )
    if existing is not None:
        raise ApplicationError(
            "benchmark_dataset_exists",
            "A dataset with this identifier and version is already registered.",
            409,
        )
    record = BenchmarkDatasetRecord(
        owner_id=owner_id,
        dataset_id=request.dataset_id,
        version=request.version,
        task=request.task.value,
        license=request.license,
        attribution=request.attribution,
        samples=[sample.model_dump(mode="json") for sample in request.samples],
        sample_count=len(request.samples),
    )
    session.add(record)
    try:
        await session.flush()
        session.add_all(
            BenchmarkDatasetAssetRecord(
                dataset_record_id=record.id,
                source_asset_id=sample.asset_id,
                sample_index=index,
            )
            for index, sample in enumerate(request.samples)
        )
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        existing = await session.scalar(
            select(BenchmarkDatasetRecord.id).where(
                BenchmarkDatasetRecord.owner_id == owner_id,
                BenchmarkDatasetRecord.dataset_id == request.dataset_id,
                BenchmarkDatasetRecord.version == request.version,
            )
        )
        if existing is not None:
            raise ApplicationError(
                "benchmark_dataset_exists",
                "A dataset with this identifier and version is already registered.",
                409,
            ) from exc
        raise
    await session.refresh(record)
    return BenchmarkDatasetResponse.model_validate(record)


@router.get(
    "/benchmarks/datasets",
    response_model=list[BenchmarkDatasetResponse],
    summary="List the caller's registered benchmark datasets",
)
async def list_benchmark_datasets(
    owner_id: OwnerId,
    session: DatabaseSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[BenchmarkDatasetResponse]:
    records = list(
        (
            await session.scalars(
                select(BenchmarkDatasetRecord)
                .where(BenchmarkDatasetRecord.owner_id == owner_id)
                .order_by(BenchmarkDatasetRecord.created_at.desc(), BenchmarkDatasetRecord.id)
                .limit(limit)
                .offset(offset)
            )
        ).all()
    )
    return [BenchmarkDatasetResponse.model_validate(record) for record in records]


@router.post(
    "/benchmarks",
    response_model=BenchmarkRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue model evaluation against a registered dataset",
)
async def queue_benchmark(
    request: BenchmarkRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> BenchmarkRunResponse:
    dataset = await session.scalar(
        select(BenchmarkDatasetRecord).where(
            BenchmarkDatasetRecord.owner_id == owner_id,
            BenchmarkDatasetRecord.dataset_id == request.dataset_id,
            BenchmarkDatasetRecord.version == request.dataset_version,
        )
    )
    if dataset is None:
        raise ApplicationError("benchmark_dataset_not_found", "Dataset was not found.", 404)
    if dataset.task != request.task.value:
        raise ApplicationError(
            "benchmark_task_mismatch",
            "The requested benchmark task does not match the dataset annotations.",
            422,
        )
    for model_id in request.model_ids:
        try:
            definition = resources.ai.registry.get(model_id)
        except ModelNotFoundError as exc:
            raise ApplicationError("model_not_found", str(exc), 404) from exc
        if request.task not in definition.supported_tasks:
            raise ApplicationError(
                "unsupported_model_task",
                f"Model '{model_id}' does not support task '{request.task.value}'.",
                422,
            )
        if not resources.ai.provider_is_configured(model_id):
            raise ApplicationError(
                "model_unavailable",
                f"Model '{model_id}' is not enabled and ready for inference.",
                503,
            )
    first_asset_id = uuid.UUID(dataset.samples[0]["asset_id"])
    try:
        job, _ = await create_job(
            session,
            owner_id,
            first_asset_id,
            "ai_benchmark",
            {
                "dataset_record_id": str(dataset.id),
                "model_ids": request.model_ids,
                "task": request.task.value,
                "parameters": request.parameters,
            },
            idempotency_key,
            commit=False,
        )
    except JobConflictError as exc:
        raise ApplicationError("job_conflict", str(exc), 409) from exc
    except ValueError as exc:
        raise ApplicationError("invalid_idempotency_key", str(exc), 422) from exc
    run = await session.scalar(
        select(BenchmarkRunRecord).where(BenchmarkRunRecord.job_id == job.id)
    )
    if run is None:
        run = BenchmarkRunRecord(
            job_id=job.id,
            dataset_record_id=dataset.id,
            owner_id=owner_id,
            model_ids=request.model_ids,
            status="queued",
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
    response.headers["Location"] = f"/api/v1/ai/benchmarks/{job.id}"
    return BenchmarkRunResponse.model_validate(run)


@router.get(
    "/benchmarks/{job_id}",
    response_model=BenchmarkRunResponse,
    summary="Get an owned benchmark run and its metrics",
)
async def get_benchmark_run(
    job_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> BenchmarkRunResponse:
    run = await session.scalar(
        select(BenchmarkRunRecord).where(
            BenchmarkRunRecord.job_id == job_id,
            BenchmarkRunRecord.owner_id == owner_id,
        )
    )
    if run is None:
        raise ApplicationError("benchmark_not_found", "Benchmark run was not found.", 404)
    return BenchmarkRunResponse.model_validate(run)
