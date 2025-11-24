"""
Hierarki exception domain dan penanganan galat aplikasi.

Modul ini menyediakan kelas-kelas exception kustom yang merepresentasikan
berbagai kondisi galat dalam alur inferensi dan validasi citra.
"""

from ..domain.constants import ErrorCode


class AppException(Exception):
    """
    Kelas dasar untuk seluruh exception domain aplikasi.

    Menyimpan kode status HTTP, kode kesalahan terstandarisasi, serta pesan
    penjelas yang akan dikembalikan kepada klien.
    """

    def __init__(self, status_code: int, code: str, message: str) -> None:
        """
        Menginisialisasi instance AppException.

        Args:
            status_code (int): Kode status HTTP yang sesuai.
            code (str): Kode unik identifikasi jenis galat.
            message (str): Pesan kesalahan ramah pengguna.
        """
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class ImageEmptyError(AppException):
    """
    Menandakan bahwa berkas citra yang diunggah tidak memiliki data atau kosong.

    Exception ini memicu respons HTTP 400 dengan kode INVALID_REQUEST.
    """

    def __init__(self, message: str = "File gambar tidak boleh kosong.") -> None:
        """
        Menginisialisasi ImageEmptyError dengan pesan kesalahan default.

        Args:
            message (str): Pesan kesalahan penjelas untuk klien.
        """
        super().__init__(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST.value,
            message=message,
        )


class PayloadTooLargeError(AppException):
    """
    Menandakan ukuran payload berkas melebihi batas maksimum yang ditentukan.

    Exception ini memicu respons HTTP 413 dengan kode PAYLOAD_TOO_LARGE.
    """

    def __init__(
        self,
        message: str = "Ukuran file melebihi batas maksimum yang diizinkan.",
    ) -> None:
        """
        Menginisialisasi PayloadTooLargeError dengan pesan batas muatan.

        Args:
            message (str): Pesan kesalahan penjelas untuk klien.
        """
        super().__init__(
            status_code=413,
            code=ErrorCode.PAYLOAD_TOO_LARGE.value,
            message=message,
        )


class UnsupportedMediaTypeError(AppException):
    """
    Menandakan format citra yang diunggah tidak didukung oleh sistem.

    Exception ini memicu respons HTTP 415 dengan kode UNSUPPORTED_MEDIA_TYPE.
    """

    def __init__(
        self,
        message: str = "Format file tidak didukung. Hanya JPEG dan PNG yang diperbolehkan.",
    ) -> None:
        """
        Menginisialisasi UnsupportedMediaTypeError dengan daftar format yang diizinkan.

        Args:
            message (str): Pesan kesalahan penjelas untuk klien.
        """
        super().__init__(
            status_code=415,
            code=ErrorCode.UNSUPPORTED_MEDIA_TYPE.value,
            message=message,
        )


class CorruptImageError(AppException):
    """
    Menandakan isi berkas rusak atau gagal didekode menjadi citra yang valid.

    Exception ini memicu respons HTTP 422 dengan kode UNPROCESSABLE_IMAGE.
    """

    def __init__(
        self,
        message: str = "File yang diunggah tidak dapat diproses sebagai gambar yang valid.",
    ) -> None:
        """
        Menginisialisasi CorruptImageError saat terjadi kegagalan dekode citra.

        Args:
            message (str): Pesan kesalahan penjelas untuk klien.
        """
        super().__init__(
            status_code=422,
            code=ErrorCode.UNPROCESSABLE_IMAGE.value,
            message=message,
        )


class ModelNotReadyError(AppException):
    """
    Menandakan artefak model belum selesai dimuat atau belum siap menerima inferensi.

    Exception ini memicu respons HTTP 503 dengan kode MODEL_UNAVAILABLE.
    """

    def __init__(
        self,
        message: str = "Model belum siap atau tidak tersedia untuk prediksi.",
    ) -> None:
        """
        Menginisialisasi ModelNotReadyError ketika kesiapan model belum tercapai.

        Args:
            message (str): Pesan kesalahan penjelas untuk klien.
        """
        super().__init__(
            status_code=503,
            code=ErrorCode.MODEL_UNAVAILABLE.value,
            message=message,
        )


class ModelContractViolationError(AppException):
    """
    Menandakan struktur atau tensor artefak model tidak memenuhi spesifikasi kontrak.

    Exception ini memicu respons HTTP 503 saat verifikasi arsitektur gagal.
    """

    def __init__(
        self,
        message: str = "Artefak model tidak sesuai dengan kontrak arsitektur atau input/output.",
    ) -> None:
        """
        Menginisialisasi ModelContractViolationError saat kontrak tensor tidak cocok.

        Args:
            message (str): Pesan kesalahan penjelas untuk klien.
        """
        super().__init__(
            status_code=503,
            code=ErrorCode.MODEL_UNAVAILABLE.value,
            message=message,
        )
