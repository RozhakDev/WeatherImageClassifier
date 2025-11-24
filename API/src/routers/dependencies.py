"""
Penyedia dependensi runtime untuk endpoint FastAPI.

Modul ini menyediakan fungsi injeksi dependensi untuk mengakses instans
layanan pemroses citra, model Keras, dan alur prediksi dari state aplikasi.
"""

from fastapi import Request

from ..services.base import (
    AbstractImageProcessor,
    AbstractModelService,
    AbstractPredictionService,
)


def get_model_service(request: Request) -> AbstractModelService:
    """
    Mengambil instans layanan model yang tersimpan di state aplikasi.

    Args:
        request (Request): Objek permintaan HTTP dari klien.

    Returns:
        AbstractModelService: Instans layanan model yang aktif.
    """
    return request.app.state.model_service


def get_image_processor(request: Request) -> AbstractImageProcessor:
    """
    Mengambil instans layanan pemrosesan citra dari state aplikasi.

    Args:
        request (Request): Objek permintaan HTTP dari klien.

    Returns:
        AbstractImageProcessor: Instans pengolah gambar yang aktif.
    """
    return request.app.state.image_processor


def get_prediction_service(request: Request) -> AbstractPredictionService:
    """
    Mengambil instans layanan alur prediksi dari state aplikasi.

    Args:
        request (Request): Objek permintaan HTTP dari klien.

    Returns:
        AbstractPredictionService: Instans layanan prediksi yang aktif.
    """
    return request.app.state.prediction_service
