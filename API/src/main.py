"""
Titik masuk utama aplikasi layanan inferensi klasifikasi citra cuaca.

Modul ini menginisialisasi aplikasi FastAPI, mendaftarkan middleware,
menangani siklus hidup layanan, dan menghubungkan router endpoint.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import settings
from .core.exceptions import AppException
from .core.logging import setup_logging
from .core.middlewares import (
    AccessLoggingMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
)
from .core.openapi import (
    OPENAPI_CONTACT,
    OPENAPI_DESCRIPTION,
    OPENAPI_LICENSE,
    OPENAPI_SUMMARY,
    OPENAPI_TAGS,
    OPENAPI_TITLE,
)
from .domain.constants import ErrorCode
from .routers.health import health_router
from .routers.predictions import predictions_router
from .services.image_processor import ImageProcessor
from .services.model_service import KerasModelService
from .services.prediction_service import PredictionService

logger = logging.getLogger("src.main")


def _build_error_response(
    status_code: int,
    code: str,
    message: str,
    instance: str,
    request_id: str,
) -> JSONResponse:
    """
    Menyusun objek respons JSON galat terstandarisasi untuk klien.

    Mengemas kode status HTTP, pesan kesalahan, identifier permintaan,
    dan timestamp UTC ke dalam format konsisten.

    Args:
        status_code (int): Kode status HTTP respons.
        code (str): Kode string unik identifikasi jenis galat.
        message (str): Pesan kesalahan ramah pengguna.
        instance (str): Path endpoint tempat terjadinya galat.
        request_id (str): Identifier unik penelusuran permintaan.

    Returns:
        JSONResponse: Objek respons JSON terstruktur dengan status code yang sesuai.
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "status": status_code,
                "type": f"https://errors.weatherclassifier.local/{code}",
                "title": code.replace("_", " ").title(),
                "instance": instance,
                "request_id": request_id,
                "timestamp": now_utc,
            }
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Mengelola siklus hidup startup dan shutdown aplikasi.

    Fungsi ini menginisialisasi logging terstruktur, memuat artefak model Keras,
    dan menyiapkan seluruh instans layanan pada state aplikasi.

    Args:
        app (FastAPI): Instans aplikasi FastAPI yang sedang berjalan.

    Yields:
        None: Menyerahkan kendali ke aplikasi selama masa aktif.
    """
    setup_logging()
    logger.info("Starting WeatherImageClassifier API service...")

    image_processor = ImageProcessor(config=settings)
    model_service = KerasModelService(config=settings)

    try:
        model_service.load_model()
        logger.info(
            f"Model loaded successfully. "
            f"Version: {model_service.model_version}"
        )
    except Exception as error:
        logger.error(
            f"Failed to load model on startup: {error}",
            exc_info=True,
        )

    prediction_service = PredictionService(
        image_processor=image_processor,
        model_service=model_service,
        config=settings,
    )

    app.state.image_processor = image_processor
    app.state.model_service = model_service
    app.state.prediction_service = prediction_service

    yield

    logger.info("Shutting down WeatherImageClassifier API service...")


def create_app() -> FastAPI:
    """
    Menginisialisasi dan mengonfigurasi instance utama aplikasi FastAPI.

    Menghubungkan middleware keamanan, exception handler kustom,
    serta mendaftarkan seluruh router endpoint layanan.

    Returns:
        FastAPI: Objek aplikasi FastAPI yang siap dijalankan.
    """
    app = FastAPI(
        title=OPENAPI_TITLE,
        summary=OPENAPI_SUMMARY,
        description=OPENAPI_DESCRIPTION,
        version=settings.app_version,
        openapi_tags=OPENAPI_TAGS,
        contact=OPENAPI_CONTACT,
        license_info=OPENAPI_LICENSE,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        openapi_url=settings.openapi_url,
        lifespan=lifespan,
    )

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(AccessLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    @app.exception_handler(AppException)
    async def handle_app_exception(
        request: Request,
        exc: AppException,
    ) -> JSONResponse:
        """
        Menangani exception domain aplikasi dan mengubahnya menjadi respons HTTP terstruktur.

        Args:
            request (Request): Objek permintaan HTTP klien.
            exc (AppException): Instance galat domain yang tertangkap.

        Returns:
            JSONResponse: Respons galat dengan format terstandarisasi.
        """
        request_id = getattr(request.state, "request_id", "")
        logger.warning(
            f"Handled application exception: {exc.code} - {exc.message}",
            extra={
                "request_id": request_id,
                "endpoint": request.url.path,
                "status_code": exc.status_code,
                "error_code": exc.code,
                "error_message": exc.message,
            },
        )
        return _build_error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            instance=request.url.path,
            request_id=request_id,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """
        Menangani galat validasi payload permintaan dari FastAPI atau Pydantic.

        Args:
            request (Request): Objek permintaan HTTP klien.
            exc (RequestValidationError): Rincian kegagalan validasi masukan.

        Returns:
            JSONResponse: Respons HTTP 400 dengan informasi galat validasi.
        """
        request_id = getattr(request.state, "request_id", "")
        logger.warning(
            f"Request validation failure on {request.url.path}: {exc.errors()}",
            extra={
                "request_id": request_id,
                "endpoint": request.url.path,
                "status_code": status.HTTP_400_BAD_REQUEST,
                "error_code": ErrorCode.INVALID_REQUEST.value,
                "validation_errors": exc.errors(),
            },
        )
        return _build_error_response(
            status_code=status.HTTP_400_BAD_REQUEST,
            code=ErrorCode.INVALID_REQUEST.value,
            message="Permintaan tidak valid atau field 'image' tidak disertakan.",
            instance=request.url.path,
            request_id=request_id,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """
        Menangani galat tak terduga pada server secara aman tanpa membocorkan trace internal.

        Args:
            request (Request): Objek permintaan HTTP klien.
            exc (Exception): Objek exception tak terduga yang terjadi.

        Returns:
            JSONResponse: Respons HTTP 500 dengan pesan kesalahan aman.
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            f"Unexpected server error for request {request_id}: {exc}",
            extra={
                "request_id": request_id,
                "endpoint": request.url.path,
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "error_code": ErrorCode.INTERNAL_SERVER_ERROR.value,
            },
            exc_info=True,
        )
        return _build_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=ErrorCode.INTERNAL_SERVER_ERROR.value,
            message="Terjadi kesalahan internal pada server.",
            instance=request.url.path,
            request_id=request_id,
        )

    app.include_router(health_router)
    app.include_router(predictions_router)

    return app


app = create_app()
