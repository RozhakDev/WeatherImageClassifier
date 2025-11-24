"""
Konfigurasi aplikasi dan manajemen variabel lingkungan.

Modul ini mendefinisikan pengaturan global yang dibutuhkan oleh layanan inferensi
dan menyediakan mekanisme parsing parameter konfigurasi.
"""

import json
from pathlib import Path
import time
from typing import Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Menyimpan seluruh konfigurasi runtime aplikasi.

    Kelas ini membaca variabel dari environment atau file .env dan
    menyediakan nilai default yang aman untuk kebutuhan inferensi citra.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "WeatherImageClassifier API"
    service_id: str = "weather-image-classifier"
    app_version: str = "1.0.0"
    model_version: str = "1.0.0"
    model_name: str = "MobileNetV2-WeatherClassifier"
    model_path: str = "models/weather_classifier.keras"
    max_upload_bytes: int = 10 * 1024 * 1024
    upload_chunk_size_bytes: int = 65536
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_formats: tuple[str, ...] | str = ("JPEG", "PNG")
    target_image_size: tuple[int, int] | str = (224, 224)
    color_mode: str = "RGB"
    color_channels: int = 3
    max_image_pixels: int = 10_000_000
    score_precision: int = 6

    docs_url: str | None = "/docs"
    redoc_url: str | None = "/redoc"
    openapi_url: str | None = "/openapi.json"

    security_hsts_header: str = "max-age=31536000; includeSubDomains"
    security_frame_options: str = "DENY"
    security_content_type_options: str = "nosniff"
    default_content_security_policy: str = "default-src 'self'"

    @field_validator("target_image_size", mode="after")
    @classmethod
    def validate_target_image_size(cls, value: Any) -> tuple[int, int]:
        """
        Memvalidasi dan mengonversi format ukuran resolusi citra target.

        Args:
            value (Any): Nilai ukuran citra dalam bentuk teks, list, atau tuple.

        Returns:
            tuple[int, int]: Pasangan dimensi tinggi dan lebar citra.
        """
        if isinstance(value, str):
            clean_str = value.strip("()[] \t\r\n")
            try:
                parts = [int(part.strip()) for part in clean_str.split(",") if part.strip()]
                if len(parts) == 2:
                    return (parts[0], parts[1])
            except (ValueError, TypeError):
                return (224, 224)
        elif isinstance(value, (list, tuple)):
            if len(value) == 2:
                try:
                    return (int(value[0]), int(value[1]))
                except (ValueError, TypeError):
                    return (224, 224)
        return (224, 224)

    @field_validator("allowed_formats", mode="after")
    @classmethod
    def validate_allowed_formats(cls, value: Any) -> tuple[str, ...]:
        """
        Memvalidasi dan mengurai daftar format berkas yang diperbolehkan.

        Args:
            value (Any): Kumpulan ekstensi format dalam bentuk JSON, teks, atau tuple.

        Returns:
            tuple[str, ...]: Koleksi format berkas gambar yang diizinkan.
        """
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return tuple(parsed)
            except Exception:
                clean_str = value.strip("()[] \t\r\n")
                parts = [
                    part.strip().strip("'\"")
                    for part in clean_str.split(",")
                    if part.strip()
                ]
                return tuple(parts)
        elif isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
        return ("JPEG", "PNG")

    def resolve_model_path(self) -> Path:
        """
        Menemukan lokasi berkas artefak model di sistem berkas.

        Returns:
            Path: Objek Path absolut atau relatif menuju berkas model.
        """
        primary_path = Path(self.model_path)
        if primary_path.is_file():
            return primary_path

        src_dir = Path(__file__).resolve().parent
        api_dir = src_dir.parent

        in_api_models = api_dir / "models" / primary_path.name
        if in_api_models.is_file():
            return in_api_models

        in_api_path = api_dir / self.model_path
        if in_api_path.is_file():
            return in_api_path

        return primary_path


START_TIME: float = time.time()
settings = Settings()
