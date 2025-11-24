"""
Konstanta domain dan enumerasi klasifikasi cuaca.

Modul ini mendefinisikan label kelas cuaca, pemetaan indeks tensor,
serta kode kesalahan terstandarisasi untuk respons API.
"""

from enum import StrEnum


class WeatherClass(StrEnum):
    """
    Kategori kelas kondisi cuaca yang didukung oleh model.

    Setiap nilai merepresentasikan kondisi cuaca spesifik hasil klasifikasi citra.
    """

    CLOUDY = "cloudy"
    FOGGY = "foggy"
    RAINY = "rainy"
    SHINE = "shine"
    SUNRISE = "sunrise"


WEATHER_CLASSES: tuple[str, ...] = (
    WeatherClass.CLOUDY.value,
    WeatherClass.FOGGY.value,
    WeatherClass.RAINY.value,
    WeatherClass.SHINE.value,
    WeatherClass.SUNRISE.value,
)

CLASS_INDEX_MAP: dict[int, str] = {
    index: class_name for index, class_name in enumerate(WEATHER_CLASSES)
}


class ErrorCode(StrEnum):
    """
    Daftar kode kesalahan terstandarisasi untuk respons galat API.

    Digunakan oleh middleware dan exception handler dalam menyusun payload error.
    """

    INVALID_REQUEST = "INVALID_REQUEST"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    UNPROCESSABLE_IMAGE = "UNPROCESSABLE_IMAGE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
