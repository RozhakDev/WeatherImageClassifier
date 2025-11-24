"""
Layanan siklus hidup dan eksekusi inferensi model machine learning.

Modul ini mengelola pemuatan berkas bobot model Keras lokal, validasi kontrak tensor,
serta eksekusi forward-pass kalkulasi skor klasifikasi cuaca.
"""

import logging
from pathlib import Path
from typing import Any
import numpy as np
import keras
import keras.layers
import keras.models

from ..config import Settings, settings as app_settings
from ..core.exceptions import ModelContractViolationError, ModelNotReadyError
from ..domain.constants import WEATHER_CLASSES
from .base import AbstractModelService

logger = logging.getLogger("src.services.model_service")


class KerasModelService(AbstractModelService):
    """
    Implementasi layanan model Keras untuk klasifikasi citra cuaca.

    Menangani pemuatan artefak model MobileNetV2 lokal, validasi bentuk tensor,
    serta prediksi skor probabilitas kelas cuaca.
    """

    def __init__(self, config: Settings = app_settings) -> None:
        """
        Menginisialisasi konfigurasi layanan model dan menerapkan patch deserialisasi.

        Args:
            config (Settings): Objek konfigurasi aplikasi.
        """
        self._config = config
        self._model: Any = None
        self._is_ready: bool = False
        self._apply_deserialization_patch()

    def _apply_deserialization_patch(self) -> None:
        """
        Menyesuaikan parameter inisialisasi layer Dense pada pustaka Keras.

        Mencegah error kompatibilitas parameter konfigurasi kuantisasi saat
        artefak model dimuat dari disk.
        """
        original_dense_init = keras.layers.Dense.__init__

        def safe_dense_init(layer_instance: Any, *args: Any, **kwargs: Any) -> None:
            kwargs.pop("quantization_config", None)
            original_dense_init(layer_instance, *args, **kwargs)

        keras.layers.Dense.__init__ = safe_dense_init

    def load_model(self) -> None:
        """
        Memuat berkas artefak model Keras dari disk ke memori.

        Memverifikasi ketersediaan berkas dan kesesuaian kontrak tensor masukan
        serta keluaran model.

        Raises:
            ModelContractViolationError: Jika berkas model tidak ditemukan atau bentuk tensor tidak valid.
        """
        resolved_path: Path = self._config.resolve_model_path()
        logger.info(f"Resolving model artifact path: {resolved_path}")
        if not resolved_path.is_file():
            self._is_ready = False
            logger.error(f"Model artifact not found at: {resolved_path}")
            raise ModelContractViolationError(
                message=f"Model artifact tidak ditemukan pada lokasi: {resolved_path}"
            )

        try:
            self._model = keras.models.load_model(str(resolved_path))
            self._validate_contract()
            self._is_ready = True
            logger.info(f"Model successfully loaded: {self.model_name} (version {self.model_version})")
        except Exception as error:
            self._is_ready = False
            self._model = None
            logger.error(f"Failed to load or validate model artifact: {error}", exc_info=True)
            raise ModelContractViolationError(
                message="Gagal memuat atau memvalidasi kontrak model."
            ) from error

    def _validate_contract(self) -> None:
        """
        Memverifikasi keselarasan bentuk tensor input dan output model dengan kontrak sistem.

        Raises:
            ModelContractViolationError: Jika dimensi tensor tidak sesuai spesifikasi.
        """
        if self._model is None:
            raise ModelContractViolationError(message="Model belum diinisialisasi.")

        input_shape = getattr(self._model, "input_shape", None)
        output_shape = getattr(self._model, "output_shape", None)

        expected_input = self.input_shape
        expected_output_units = len(WEATHER_CLASSES)

        if input_shape != expected_input:
            logger.error(f"Input contract mismatch: got {input_shape}, expected {expected_input}")
            raise ModelContractViolationError(
                message=f"Bentuk input {input_shape} tidak sesuai kontrak {expected_input}."
            )

        if not output_shape or output_shape[-1] != expected_output_units:
            logger.error(f"Output contract mismatch: got {output_shape}, expected {expected_output_units} classes")
            raise ModelContractViolationError(
                message=f"Bentuk output {output_shape} tidak memiliki {expected_output_units} unit kelas."
            )

        logger.info(
            "Model contract verified successfully",
            extra={
                "input_shape": list(input_shape),
                "output_shape": list(output_shape),
                "num_classes": expected_output_units,
            },
        )

    def predict(self, preprocessed_batch: np.ndarray) -> np.ndarray:
        """
        Menjalankan kalkulasi inferensi forward-pass menggunakan model Keras.

        Args:
            preprocessed_batch (np.ndarray): Batch tensor citra yang telah dinormalisasi.

        Returns:
            np.ndarray: Array nilai probabilitas untuk seluruh kelas cuaca.

        Raises:
            ModelNotReadyError: Jika model belum dimuat atau belum siap menerima permintaan.
        """
        if not self._is_ready or self._model is None:
            logger.warning("Predict called while model is not ready")
            raise ModelNotReadyError()

        logger.debug(
            "Executing forward-pass inference",
            extra={"batch_shape": list(preprocessed_batch.shape)},
        )
        raw_predictions = self._model.predict(preprocessed_batch, verbose=0)
        return np.asarray(raw_predictions, dtype=np.float32)

    @property
    def is_ready(self) -> bool:
        """
        Status kesiapan artefak model untuk menerima inferensi.

        Returns:
            bool: Nilai True jika model aktif dan telah terverifikasi.
        """
        return self._is_ready

    @property
    def model_version(self) -> str:
        """
        Versi rilis dari artefak model yang aktif.

        Returns:
            str: Versi model sesuai konfigurasi.
        """
        return self._config.model_version

    @property
    def model_name(self) -> str:
        """
        Nama arsitektur model machine learning.

        Returns:
            str: Nama model sesuai konfigurasi.
        """
        return self._config.model_name

    @property
    def input_shape(self) -> tuple[int | None, ...]:
        """
        Bentuk dimensi tensor masukan yang diharapkan oleh model.

        Returns:
            tuple[int | None, ...]: Dimensi batch, tinggi, lebar, dan kanal warna citra.
        """
        return (
            None,
            self._config.target_image_size[0],
            self._config.target_image_size[1],
            self._config.color_channels,
        )

    @property
    def device(self) -> str:
        """
        Perangkat komputasi tempat inferensi dijalankan.

        Returns:
            str: Tipe perangkat komputasi aktif (misalnya CPU).
        """
        return "CPU"
