"""
Endpoint pemeriksaan kesehatan dan status operasional sistem.

Modul ini menyediakan probe liveness untuk memastikan proses berjalan,
serta probe readiness untuk memverifikasi kesiapan model inferensi.
"""

from datetime import datetime, timezone
import logging
import time
from fastapi import APIRouter, Depends, status

from ..config import START_TIME, settings
from ..core.exceptions import ModelNotReadyError
from ..core.openapi import (
    LIVENESS_ENDPOINT_DESCRIPTION,
    LIVENESS_ENDPOINT_SUMMARY,
    LIVENESS_RESPONSES,
    READINESS_ENDPOINT_DESCRIPTION,
    READINESS_ENDPOINT_SUMMARY,
    READINESS_RESPONSES,
)
from ..domain.constants import WEATHER_CLASSES
from ..domain.schemas import (
    EngineCheckDetail,
    LivenessResponse,
    ModelCheckDetail,
    ReadinessChecks,
    ReadinessResponse,
)
from ..services.base import AbstractModelService
from .dependencies import get_model_service

logger = logging.getLogger("src.routers.health")
health_router = APIRouter(prefix="/health", tags=["Status Layanan"])


def _get_current_utc_timestamp() -> str:
    """
    Mengambil waktu saat ini dalam format string standar UTC ISO 8601.

    Returns:
        str: Representasi waktu UTC terkini.
    """
    return datetime.now(timezone.utc).isoformat()


def _get_service_uptime() -> float:
    """
    Menghitung total durasi waktu server aktif berjalan sejak startup.

    Returns:
        float: Waktu aktif sistem dalam satuan detik.
    """
    return round(time.time() - START_TIME, 2)


@health_router.get(
    "/live",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary=LIVENESS_ENDPOINT_SUMMARY,
    description=LIVENESS_ENDPOINT_DESCRIPTION,
    response_description="Status liveness proses server HTTP.",
    responses=LIVENESS_RESPONSES,
)
async def check_liveness() -> LivenessResponse:
    """
    Memeriksa kelangsungan hidup proses aplikasi server HTTP.

    Endpoint ini merespons dengan cepat untuk menandakan bahwa kontainer
    atau proses aplikasi masih hidup dan mampu menerima koneksi.

    Returns:
        LivenessResponse: Status kelangsungan proses aplikasi.
    """
    return LivenessResponse(
        status="pass",
        service_id=settings.service_id,
        version=settings.app_version,
        timestamp=_get_current_utc_timestamp(),
        uptime_seconds=_get_service_uptime(),
    )


@health_router.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary=READINESS_ENDPOINT_SUMMARY,
    description=READINESS_ENDPOINT_DESCRIPTION,
    response_description="Status kesiapan operasional model dan engine inferensi.",
    responses=READINESS_RESPONSES,
)
async def check_readiness(
    model_service: AbstractModelService = Depends(get_model_service),
) -> ReadinessResponse:
    """
    Memeriksa kesiapan model machine learning dan mesin inferensi.

    Endpoint ini memastikan artefak model telah berhasil dimuat ke memori
    dan siap menerima permintaan inferensi prediksi citra.

    Args:
        model_service (AbstractModelService): Instans layanan model dari dependensi aplikasi.

    Returns:
        ReadinessResponse: Rincian status kesiapan artefak dan komponen inferensi.

    Raises:
        ModelNotReadyError: Jika model belum dimuat atau belum siap beroperasi.
    """
    if not model_service.is_ready:
        logger.warning("Readiness probe check failed: model is not ready")
        raise ModelNotReadyError()

    checks = ReadinessChecks(
        model=ModelCheckDetail(
            status="pass",
            model_name=model_service.model_name,
            model_version=model_service.model_version,
            input_shape=model_service.input_shape,
            num_classes=len(WEATHER_CLASSES),
            classes=WEATHER_CLASSES,
            framework="Keras 3 / TensorFlow",
            is_ready=model_service.is_ready,
        ),
        inference_engine=EngineCheckDetail(
            status="pass",
            device=model_service.device,
        ),
    )

    return ReadinessResponse(
        status="pass",
        service_id=settings.service_id,
        version=settings.app_version,
        timestamp=_get_current_utc_timestamp(),
        uptime_seconds=_get_service_uptime(),
        checks=checks,
    )
