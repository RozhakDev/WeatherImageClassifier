"""
Antarmuka abstrak (kontrak) untuk layanan pemrosesan dan inferensi.

Modul ini mendefinisikan kelas basis abstrak yang menetapkan standar implementasi
komponen prapemrosesan citra, manajemen model machine learning, serta orkestrasi prediksi.
"""

from abc import ABC, abstractmethod
import numpy as np

from ..domain.schemas import InputImageMetadata, PredictionResponse


class AbstractImageProcessor(ABC):
    """
    Kontrak antarmuka untuk komponen prapemrosesan citra.

    Menjamin setiap implementasi pengolah gambar mampu mengonversi berkas biner
    menjadi batch tensor yang siap dikonsumsi model.
    """

    @abstractmethod
    def process_image(
        self,
        file_bytes: bytes,
    ) -> tuple[np.ndarray, InputImageMetadata]:
        """
        Memproses berkas biner gambar menjadi tensor dan metadata.

        Args:
            file_bytes (bytes): Konten byte mentah dari berkas citra.

        Returns:
            tuple[np.ndarray, InputImageMetadata]: Pasangan array tensor gambar dan objek metadata.
        """
        raise NotImplementedError


class AbstractModelService(ABC):
    """
    Kontrak antarmuka untuk manajemen siklus hidup dan inferensi model.

    Menetapkan metode pemuatan artefak, eksekusi forward-pass, serta atribut
    diagnostik kesiapan model machine learning.
    """

    @abstractmethod
    def load_model(self) -> None:
        """
        Memuat artefak model ke memori dan memverifikasi kontrak arsitekturnya.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, preprocessed_batch: np.ndarray) -> np.ndarray:
        """
        Menjalankan kalkulasi inferensi pada sekumpulan data citra.

        Args:
            preprocessed_batch (np.ndarray): Batch tensor citra yang telah dinormalisasi.

        Returns:
            np.ndarray: Array skor kemungkinan hasil inferensi model.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """
        Memeriksa apakah artefak model telah berhasil dimuat dan siap melayani prediksi.

        Returns:
            bool: True jika model telah siap digunakan.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_version(self) -> str:
        """
        Versi rilis dari artefak model yang aktif.

        Returns:
            str: Teks nomor versi model.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Nama arsitektur atau identitas model machine learning.

        Returns:
            str: Teks nama model.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int | None, ...]:
        """
        Dimensi tensor masukan yang diharapkan oleh arsitektur model.

        Returns:
            tuple[int | None, ...]: Dimensi tinggi, lebar, dan kanal warna citra.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> str:
        """
        Nama perangkat keras komputasi tempat inferensi dijalankan.

        Returns:
            str: Tipe perangkat komputasi, misalnya CPU.
        """
        raise NotImplementedError


class AbstractPredictionService(ABC):
    """
    Kontrak antarmuka untuk alur orkestrasi inferensi menyeluruh.

    Menghubungkan tahapan prapemrosesan citra, kalkulasi model, serta penyusunan respons hasil prediksi.
    """

    @abstractmethod
    def predict(
        self,
        file_bytes: bytes,
        request_id: str = "",
    ) -> PredictionResponse:
        """
        Menjalankan alur inferensi lengkap dari berkas mentah hingga respons prediksi.

        Args:
            file_bytes (bytes): Konten byte mentah dari berkas citra.
            request_id (str): Identifier unik permintaan untuk pelacakan log.

        Returns:
            PredictionResponse: Objek hasil klasifikasi citra cuaca beserta metrik latensi.
        """
        raise NotImplementedError
