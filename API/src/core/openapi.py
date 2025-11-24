"""
Definisi metadata dan skema OpenAPI untuk dokumentasi layanan.

Modul ini memuat konstanta deskripsi, lisensi, informasi kontak,
serta spesifikasi respons HTTP untuk antarmuka Swagger dan ReDoc.
"""

from typing import Any
from fastapi import status

from ..domain.schemas import (
    ErrorResponse,
    LivenessResponse,
    PredictionResponse,
    ReadinessResponse,
)

OPENAPI_TITLE: str = "WeatherImageClassifier API"

OPENAPI_SUMMARY: str = "Layanan HTTP berbasis FastAPI untuk klasifikasi citra cuaca (cloudy, foggy, rainy, shine, sunrise) menggunakan model MobileNetV2."

OPENAPI_DESCRIPTION: str = (
    "## Ruang Lingkup Layanan\n\n"
    "Layanan ini menerima berkas gambar melalui HTTP dan mengklasifikasikan kondisi cuaca ke dalam salah satu dari 5 kelas: `cloudy` (mendung), `foggy` (berkabut), `rainy` (hujan), `shine` (cerah), atau `sunrise` (matahari terbit).\n\n"
    "## Kontrak & Alur Inferensi Model\n\n"
    "- **Artefak Model:** Menggunakan model Keras lokal yang dimuat dari `models/weather_classifier.keras`.\n"
    "- **Resolusi & Kanal:** Citra masukan didekode ke RGB lalu diubah menjadi ukuran `224 × 224` piksel.\n"
    "- **Normalisasi:** Transformasi nilai piksel menggunakan `mobilenet_v2.preprocess_input`.\n"
    "- **Output Model:** Menghasilkan lima skor softmax dengan pemetaan indeks tetap: `0=cloudy`, `1=foggy`, `2=rainy`, `3=shine`, `4=sunrise`.\n\n"
    "## Ketentuan Berkas & Keamanan Input\n\n"
    "- **Format yang Didukung:** Berkas citra berekstensi `JPEG` dan `PNG` dengan verifikasi isi byte melalui Pillow.\n"
    "- **Batas Ukuran:** Maksimum 10 MB per berkas (konfigurasi `MAX_UPLOAD_BYTES`).\n"
    "- **Penyimpanan:** Citra diproses di memori dan tidak disimpan di disk server.\n\n"
    "## Pemeriksaan Status Sistem\n\n"
    "- `/health/live`: Memverifikasi proses aplikasi berjalan aktif.\n"
    "- `/health/ready`: Memverifikasi bahwa model telah dimuat dan siap menerima permintaan prediksi.\n"
)

OPENAPI_TAGS: list[dict[str, Any]] = [
    {
        "name": "Prediksi Cuaca",
        "description": "Endpoint pengiriman berkas citra untuk klasifikasi kondisi cuaca secara langsung.",
    },
    {
        "name": "Status Layanan",
        "description": "Endpoint diagnostik untuk memeriksa kelangsungan proses aplikasi dan kesiapan model.",
    },
]

OPENAPI_CONTACT: dict[str, str] = {
    "name": "Tim Pengembang WeatherImageClassifier",
    "url": "https://github.com/WeatherImageClassifier",
    "email": "contact@rozhak.xyz",
}

OPENAPI_LICENSE: dict[str, str] = {
    "name": "Apache-2.0",
    "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
}

PREDICTION_ENDPOINT_SUMMARY: str = "Klasifikasi citra cuaca"

PREDICTION_ENDPOINT_DESCRIPTION: str = (
    "Menerima berkas citra melalui form field 'image', memvalidasi konten byte gambar (JPEG/PNG), "
    "mengubah ukuran ke 224x224 RGB, dan mengembalikan kelas cuaca dengan skor tertinggi beserta skor untuk kelima kelas."
)

PREDICTION_FILE_DESCRIPTION: str = "Berkas gambar cuaca berformat JPEG atau PNG (maksimum 10 MB)."

PREDICTION_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_200_OK: {
        "model": PredictionResponse,
        "description": "Prediksi berhasil. Mengembalikan label cuaca, nilai skor softmax, distribusi skor seluruh kelas, dan durasi proses.",
    },
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "Permintaan tidak valid, field 'image' tidak disertakan, atau berkas kosong.",
    },
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "model": ErrorResponse,
        "description": "Ukuran berkas melebihi batas konfigurasi MAX_UPLOAD_BYTES (10 MB).",
    },
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
        "model": ErrorResponse,
        "description": "Format berkas tidak didukung. Layanan hanya menerima format JPEG dan PNG.",
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": "Isi berkas rusak atau tidak dapat didekode sebagai citra yang valid.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "Terjadi kesalahan internal pada server saat menjalankan inferensi.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "Model belum siap atau belum selesai dimuat ke memori.",
    },
}

LIVENESS_ENDPOINT_SUMMARY: str = "Pemeriksaan proses aplikasi"

LIVENESS_ENDPOINT_DESCRIPTION: str = "Menandakan bahwa proses aplikasi berjalan dan dapat menerima koneksi. Endpoint ini tidak menjalankan inferensi atau memeriksa model."

LIVENESS_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_200_OK: {
        "model": LivenessResponse,
        "description": "Proses aplikasi berjalan normal.",
    },
}

READINESS_ENDPOINT_SUMMARY: str = "Pemeriksaan kesiapan model"

READINESS_ENDPOINT_DESCRIPTION: str = "Menandakan bahwa model Keras telah selesai dimuat, kontrak tensor terverifikasi, dan layanan siap menerima permintaan prediksi."

READINESS_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_200_OK: {
        "model": ReadinessResponse,
        "description": "Model siap melayani prediksi.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "Terjadi kesalahan internal saat memeriksa kesiapan model.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "Model belum siap, gagal dimuat, atau validasi kontrak tidak terpenuhi.",
    },
}
