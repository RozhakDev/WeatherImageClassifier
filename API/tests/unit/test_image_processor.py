"""
Pengujian unit untuk komponen pemrosesan dan dekode citra.

Memverifikasi dekode format JPEG dan PNG, ekstraksi metadata,
serta penolakan berkas rusak atau format yang tidak didukung.
"""

import pytest

from src.config import settings
from src.core.exceptions import CorruptImageError, UnsupportedMediaTypeError
from src.domain.schemas import InputImageMetadata
from src.services.image_processor import ImageProcessor


@pytest.mark.unit
class TestImageProcessorUnit:
    """
    Rangkaian pengujian unit untuk kelas ImageProcessor.

    Memastikan transformasi berkas biner citra menjadi tensor float32
    dan penanganan galat format berkas berjalan sesuai standar.
    """

    @pytest.fixture
    def processor(self) -> ImageProcessor:
        """
        Menyediakan instance ImageProcessor yang terhubung dengan pengaturan default.

        Returns:
            ImageProcessor: Objek pemroses citra untuk pengujian.
        """
        return ImageProcessor(config=settings)

    def test_preprocess_valid_jpeg(
        self,
        processor: ImageProcessor,
        valid_jpeg_bytes: bytes,
    ) -> None:
        """
        Memverifikasi bahwa citra JPEG valid berhasil diubah menjadi batch tensor 224x224x3.

        Args:
            processor (ImageProcessor): Objek pemroses citra.
            valid_jpeg_bytes (bytes): Konten biner citra JPEG valid.
        """
        batch, metadata = processor.process_image(valid_jpeg_bytes)
        assert batch.shape == (1, 224, 224, 3)
        assert isinstance(metadata, InputImageMetadata)
        assert metadata.format == "JPEG"
        assert metadata.channels == 3
        assert metadata.file_size_bytes == len(valid_jpeg_bytes)

    def test_preprocess_valid_png(
        self,
        processor: ImageProcessor,
        valid_png_bytes: bytes,
    ) -> None:
        """
        Memverifikasi bahwa citra PNG valid berhasil dikonversi ke RGB dan batch tensor 224x224x3.

        Args:
            processor (ImageProcessor): Objek pemroses citra.
            valid_png_bytes (bytes): Konten biner citra PNG valid.
        """
        batch, metadata = processor.process_image(valid_png_bytes)
        assert batch.shape == (1, 224, 224, 3)
        assert isinstance(metadata, InputImageMetadata)
        assert metadata.format == "PNG"
        assert metadata.channels == 3

    def test_preprocess_unsupported_format_raises_error(
        self,
        processor: ImageProcessor,
        valid_gif_bytes: bytes,
    ) -> None:
        """
        Memverifikasi bahwa format citra selain JPEG dan PNG memicu UnsupportedMediaTypeError.

        Args:
            processor (ImageProcessor): Objek pemroses citra.
            valid_gif_bytes (bytes): Konten biner citra GIF.
        """
        with pytest.raises(UnsupportedMediaTypeError):
            processor.process_image(valid_gif_bytes)

    def test_preprocess_corrupt_bytes_raises_error(
        self,
        processor: ImageProcessor,
        corrupt_bytes: bytes,
    ) -> None:
        """
        Memverifikasi bahwa data biner citra rusak memicu CorruptImageError.

        Args:
            processor (ImageProcessor): Objek pemroses citra.
            corrupt_bytes (bytes): Rangkaian byte acak yang rusak.
        """
        with pytest.raises(CorruptImageError):
            processor.process_image(corrupt_bytes)
