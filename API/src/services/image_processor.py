"""
Layanan pengolahan citra dan prapemrosesan tensor.

Modul ini bertanggung jawab memeriksa integritas berkas citra, mengubah ukuran piksel,
dan melakukan normalisasi masukan sesuai kebutuhan model MobileNetV2.
"""

import io
import logging
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from ..config import Settings, settings as app_settings
from ..core.exceptions import (
    CorruptImageError,
    ImageEmptyError,
    UnsupportedMediaTypeError,
)
from ..domain.schemas import InputImageMetadata
from .base import AbstractImageProcessor

logger = logging.getLogger("src.services.image_processor")


class ImageProcessor(AbstractImageProcessor):
    """
    Komponen pemroses dan penormalisasi berkas citra masukan.

    Menangani konversi biner citra menjadi tensor float32 yang kompatibel
    dengan spesifikasi input model klasifikasi cuaca.
    """

    def __init__(self, config: Settings = app_settings) -> None:
        """
        Menginisialisasi konfigurasi batas piksel dan ukuran target citra.

        Args:
            config (Settings): Objek pengaturan konfigurasi aplikasi.
        """
        self._config = config
        Image.MAX_IMAGE_PIXELS = self._config.max_image_pixels

    def process_image(self, file_bytes: bytes) -> tuple[np.ndarray, InputImageMetadata]:
        """
        Mendekode berkas biner, memvalidasi format, serta menormalisasi piksel citra.

        Args:
            file_bytes (bytes): Konten byte mentah dari berkas citra yang diunggah.

        Returns:
            tuple[np.ndarray, InputImageMetadata]: Batch tensor hasil prapemrosesan dan metadata teknis citra.

        Raises:
            ImageEmptyError: Jika byte citra bernilai kosong.
            UnsupportedMediaTypeError: Jika format citra bukan JPEG atau PNG.
            CorruptImageError: Jika berkas rusak dan tidak dapat didekode oleh Pillow.
        """
        if not file_bytes:
            logger.warning("Image processing rejected: empty byte buffer")
            raise ImageEmptyError()

        try:
            with Image.open(io.BytesIO(file_bytes)) as image:
                image_format = image.format
                if image_format not in self._config.allowed_formats:
                    logger.warning(
                        f"Unsupported image format: {image_format}",
                        extra={
                            "detected_format": image_format,
                            "allowed_formats": list(self._config.allowed_formats),
                            "file_size_bytes": len(file_bytes),
                        },
                    )
                    raise UnsupportedMediaTypeError()

                metadata = InputImageMetadata(
                    width=image.width,
                    height=image.height,
                    channels=self._config.color_channels,
                    format=str(image_format),
                    file_size_bytes=len(file_bytes),
                )

                image_rgb = image.convert(self._config.color_mode)
                resized_image = image_rgb.resize(self._config.target_image_size, Image.Resampling.BILINEAR)
                image_array = np.array(resized_image, dtype=np.float32)
                batch_array = np.expand_dims(image_array, axis=0)

                logger.debug(
                    "Image decoded and preprocessed successfully",
                    extra={
                        "original_width": image.width,
                        "original_height": image.height,
                        "target_size": list(self._config.target_image_size),
                        "color_mode": self._config.color_mode,
                        "format": str(image_format),
                    },
                )

                return preprocess_input(batch_array), metadata
        except (UnidentifiedImageError, OSError) as error:
            logger.warning(
                f"Image decoding failed: {error}",
                extra={
                    "error_type": type(error).__name__,
                    "error_detail": str(error),
                    "file_size_bytes": len(file_bytes),
                },
            )
            raise CorruptImageError() from error
