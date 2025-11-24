"""
Endpoint inferensi untuk klasifikasi citra kondisi cuaca.

Modul ini mengelola penerimaan unggahan berkas gambar melalui HTTP POST,
validasi ukuran muatan, serta penerusan citra ke alur prediksi.
"""

import logging
from fastapi import APIRouter, Depends, File, Request, UploadFile, status

from ..config import settings
from ..core.exceptions import ImageEmptyError, PayloadTooLargeError
from ..core.openapi import (
    PREDICTION_ENDPOINT_DESCRIPTION,
    PREDICTION_ENDPOINT_SUMMARY,
    PREDICTION_FILE_DESCRIPTION,
    PREDICTION_RESPONSES,
)
from ..domain.schemas import PredictionResponse
from ..services.base import AbstractPredictionService
from .dependencies import get_prediction_service

logger = logging.getLogger("src.routers.predictions")
predictions_router = APIRouter(
    prefix="/v1/predictions",
    tags=["Prediksi Cuaca"],
)


async def _read_upload_stream(
    image: UploadFile,
    chunk_size: int,
    max_bytes: int,
    request_id: str,
) -> bytes:
    """
    Membaca aliran byte berkas yang diunggah secara bertahap (chunked).

    Memastikan konsumsi memori tetap terkendali dan membatalkan proses
    jika berkas kosong atau ukurannya melebihi kuota maksimum.

    Args:
        image (UploadFile): Objek berkas unggahan dari permintaan klien.
        chunk_size (int): Ukuran setiap pecahan byte yang dibaca per iterasi.
        max_bytes (int): Ambang batas ukuran berkas maksimum yang diizinkan.
        request_id (str): Identifier unik permintaan untuk pencatatan log.

    Returns:
        bytes: Data biner utuh dari berkas citra yang diunggah.

    Raises:
        PayloadTooLargeError: Jika ukuran berkas melampaui batas maksimum.
        ImageEmptyError: Jika berkas yang diunggah tidak memiliki data atau kosong.
    """
    total_read = 0
    buffer = bytearray()

    while True:
        chunk = await image.read(chunk_size)
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > max_bytes:
            logger.warning(
                "Upload rejected: payload size exceeds maximum limit",
                extra={
                    "request_id": request_id,
                    "bytes_read": total_read,
                    "max_upload_bytes": max_bytes,
                },
            )
            raise PayloadTooLargeError()
        buffer.extend(chunk)

    if total_read == 0:
        logger.warning(
            "Upload rejected: image file is empty",
            extra={"request_id": request_id},
        )
        raise ImageEmptyError()

    return bytes(buffer)


@predictions_router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary=PREDICTION_ENDPOINT_SUMMARY,
    description=PREDICTION_ENDPOINT_DESCRIPTION,
    response_description="Hasil inferensi klasifikasi citra cuaca beserta metrik latensi dan metadata citra.",
    responses=PREDICTION_RESPONSES,
)
async def predict_weather_image(
    request: Request,
    image: UploadFile = File(..., description=PREDICTION_FILE_DESCRIPTION),
    prediction_service: AbstractPredictionService = Depends(
        get_prediction_service
    ),
) -> PredictionResponse:
    """
    Menangani pengiriman berkas citra untuk klasifikasi kondisi cuaca.

    Menerima berkas gambar dari form-data, memverifikasi aliran byte,
    dan mengembalikan label prediksi cuaca beserta rincian skor.

    Args:
        request (Request): Objek permintaan HTTP dari klien.
        image (UploadFile): Berkas gambar yang diunggah melalui formulir.
        prediction_service (AbstractPredictionService): Layanan orkestrator inferensi cuaca.

    Returns:
        PredictionResponse: Objek hasil klasifikasi cuaca, metrik waktu, dan metadata citra.
    """
    request_id: str = getattr(request.state, "request_id", "")
    logger.debug(
        "Received image upload for prediction",
        extra={
            "request_id": request_id,
            "filename": image.filename,
            "content_type": image.content_type,
        },
    )

    file_bytes = await _read_upload_stream(
        image=image,
        chunk_size=settings.upload_chunk_size_bytes,
        max_bytes=settings.max_upload_bytes,
        request_id=request_id,
    )
    return prediction_service.predict(file_bytes, request_id=request_id)
