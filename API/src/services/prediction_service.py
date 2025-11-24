"""
Layanan alur prediksi dan agregasi hasil inferensi citra cuaca.

Modul ini mengorkestrasi prapemrosesan citra, forward-pass model Keras,
perhitungan latensi, dan pemetaan skor ke format respons prediksi.
"""

from datetime import datetime, timezone
import logging
import time
import numpy as np

from ..config import Settings, settings as app_settings
from ..core.exceptions import ModelNotReadyError
from ..domain.constants import CLASS_INDEX_MAP
from ..domain.schemas import PredictionResponse, TimingDetail
from .base import (
    AbstractImageProcessor,
    AbstractModelService,
    AbstractPredictionService,
)

logger = logging.getLogger("src.services.prediction_service")


class PredictionService(AbstractPredictionService):
    """
    Orkestrator alur inferensi klasifikasi citra cuaca.

    Menghubungkan komponen pengolah citra dan layanan model untuk menghasilkan
    output prediksi yang lengkap beserta metrik kinerja.
    """

    def __init__(
        self,
        image_processor: AbstractImageProcessor,
        model_service: AbstractModelService,
        config: Settings = app_settings,
    ) -> None:
        """
        Menginisialisasi dependensi layanan pemrosesan citra dan model.

        Args:
            image_processor (AbstractImageProcessor): Komponen pengolah gambar.
            model_service (AbstractModelService): Komponen inferensi model.
            config (Settings): Objek konfigurasi aplikasi.
        """
        self._image_processor = image_processor
        self._model_service = model_service
        self._config = config

    def predict(
        self,
        file_bytes: bytes,
        request_id: str = "",
    ) -> PredictionResponse:
        """
        Mengeksekusi alur prediksi lengkap dari data byte citra mentah.

        Args:
            file_bytes (bytes): Konten byte berkas citra masukan.
            request_id (str): Identifier unik permintaan untuk penelusuran log.

        Returns:
            PredictionResponse: Objek hasil prediksi cuaca lengkap dengan metrik latensi.

        Raises:
            ModelNotReadyError: Jika model belum siap melayani inferensi.
        """
        if not self._model_service.is_ready:
            logger.warning(
                "Prediction rejected: model service is not ready",
                extra={"request_id": request_id},
            )
            raise ModelNotReadyError()

        preprocess_start = time.perf_counter()
        preprocessed_batch, metadata = self._image_processor.process_image(file_bytes)
        preprocess_ms = round((time.perf_counter() - preprocess_start) * 1000, 2)

        inference_start = time.perf_counter()
        prediction_batch = self._model_service.predict(preprocessed_batch)
        inference_ms = round((time.perf_counter() - inference_start) * 1000, 2)

        total_ms = round(preprocess_ms + inference_ms, 2)
        scores = prediction_batch[0]

        top_index = int(np.argmax(scores))
        predicted_class = CLASS_INDEX_MAP[top_index]
        class_scores: dict[str, float] = {
            class_name: round(float(scores[index]), self._config.score_precision)
            for index, class_name in CLASS_INDEX_MAP.items()
        }
        top_score = class_scores[predicted_class]
        timestamp = datetime.now(timezone.utc).isoformat()

        timing = TimingDetail(
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            total_ms=total_ms,
        )

        logger.info(
            "Prediction pipeline completed successfully",
            extra={
                "request_id": request_id,
                "predicted_class": predicted_class,
                "class_id": top_index,
                "confidence": top_score,
                "latency_total_ms": total_ms,
                "latency_preprocess_ms": preprocess_ms,
                "latency_inference_ms": inference_ms,
                "model_version": self._model_service.model_version,
            },
        )

        return PredictionResponse(
            predicted_class=predicted_class,
            class_id=top_index,
            score=top_score,
            class_scores=class_scores,
            model_version=self._model_service.model_version,
            request_id=request_id,
            timestamp=timestamp,
            timing=timing,
            input_metadata=metadata,
        )
