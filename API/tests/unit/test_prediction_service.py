"""
Pengujian unit untuk alur orkestrasi layanan prediksi.

Memverifikasi kalkulasi probabilitas kelas cuaca, perhitungan durasi proses,
serta proteksi ketika model machine learning belum siap.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.config import settings
from src.core.exceptions import ModelNotReadyError
from src.domain.constants import WEATHER_CLASSES
from src.domain.schemas import InputImageMetadata, PredictionResponse
from src.services.prediction_service import PredictionService


@pytest.mark.unit
class TestPredictionServiceUnit:
    """
    Rangkaian pengujian unit untuk kelas PredictionService.

    Menguji integrasi tiruan (mock) antara komponen pengolah gambar
    dan layanan model dalam alur prediksi cuaca.
    """

    @pytest.fixture
    def mock_image_processor(self) -> MagicMock:
        """
        Menyediakan objek tiruan (mock) dari AbstractImageProcessor.

        Returns:
            MagicMock: Mock image processor yang menghasilkan batch tensor dummy.
        """
        processor = MagicMock()
        dummy_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
        dummy_metadata = InputImageMetadata(
            width=224,
            height=224,
            channels=3,
            format="JPEG",
            file_size_bytes=1024,
        )
        processor.process_image.return_value = (dummy_batch, dummy_metadata)
        return processor

    @pytest.fixture
    def mock_model_service(self) -> MagicMock:
        """
        Menyediakan objek tiruan (mock) dari AbstractModelService.

        Returns:
            MagicMock: Mock model service dengan skor probabilitas dummy.
        """
        service = MagicMock()
        service.is_ready = True
        service.model_version = "1.0.0"
        service.predict.return_value = np.array(
            [[0.01, 0.02, 0.03, 0.90, 0.04]], dtype=np.float32
        )
        return service

    @pytest.fixture
    def service(
        self,
        mock_image_processor: MagicMock,
        mock_model_service: MagicMock,
    ) -> PredictionService:
        """
        Menyediakan instans PredictionService dengan dependensi mock.

        Args:
            mock_image_processor (MagicMock): Komponen mock pengolah gambar.
            mock_model_service (MagicMock): Komponen mock layanan model.

        Returns:
            PredictionService: Objek layanan prediksi siap uji.
        """
        return PredictionService(
            image_processor=mock_image_processor,
            model_service=mock_model_service,
            config=settings,
        )

    def test_predict_success(
        self,
        service: PredictionService,
        valid_jpeg_bytes: bytes,
    ) -> None:
        """
        Memverifikasi bahwa alur prediksi berhasil menghasilkan PredictionResponse lengkap.

        Args:
            service (PredictionService): Instans layanan prediksi.
            valid_jpeg_bytes (bytes): Data biner citra valid.
        """
        result = service.predict(valid_jpeg_bytes, request_id="custom-uuid-123")
        assert isinstance(result, PredictionResponse)
        assert result.predicted_class == "shine"
        assert result.class_id == 3
        assert result.score == 0.9
        assert len(result.class_scores) == len(WEATHER_CLASSES)
        assert result.class_scores["shine"] == 0.9
        assert result.request_id == "custom-uuid-123"
        assert result.timing.total_ms >= 0.0

    def test_predict_raises_when_model_not_ready(
        self,
        mock_image_processor: MagicMock,
        mock_model_service: MagicMock,
        valid_jpeg_bytes: bytes,
    ) -> None:
        """
        Memverifikasi bahwa alur prediksi memicu ModelNotReadyError jika model belum siap.

        Args:
            mock_image_processor (MagicMock): Komponen mock pengolah gambar.
            mock_model_service (MagicMock): Komponen mock layanan model.
            valid_jpeg_bytes (bytes): Data biner citra valid.
        """
        mock_model_service.is_ready = False
        service = PredictionService(
            image_processor=mock_image_processor,
            model_service=mock_model_service,
            config=settings,
        )
        with pytest.raises(ModelNotReadyError):
            service.predict(valid_jpeg_bytes, request_id="fail-id")
