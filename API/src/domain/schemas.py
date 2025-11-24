"""
Skema data Pydantic untuk validasi payload permintaan dan respons API.

Modul ini mendefinisikan model data terstruktur yang digunakan untuk kontrak
antarmuka inferensi, pelaporan kesehatan layanan, serta format pesan galat.
"""

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """
    Rincian informasi galat yang mematuhi format RFC 7807 / 9457.

    Menyediakan konteks teknis lengkap mengenai kegagalan pemrosesan permintaan klien.
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        ...,
        description="Kode identifikasi jenis kesalahan",
        examples=["INVALID_REQUEST"],
    )
    message: str = Field(
        ...,
        description="Pesan penjelasan kesalahan untuk pengguna",
        examples=["Permintaan tidak valid."],
    )
    status: int = Field(
        default=400,
        description="Kode status HTTP numerik",
        examples=[400],
    )
    type: str = Field(
        default="about:blank",
        description="Tautan referensi tipe kesalahan",
        examples=[
            "https://errors.weatherclassifier.local/INVALID_REQUEST"
        ],
    )
    title: str = Field(
        default="Error",
        description="Judul ringkas tipe kesalahan",
        examples=["Bad Request"],
    )
    instance: str = Field(
        default="",
        description="Path endpoint yang menghasilkan kesalahan",
        examples=["/v1/predictions"],
    )
    request_id: str = Field(
        default="",
        description="ID penelusuran permintaan",
        examples=["c4e5cb8b-2794-4c12-8f4b-fadf3b941ca2"],
    )
    timestamp: str = Field(
        default="",
        description="Waktu kejadian dalam format UTC ISO 8601",
        examples=["2026-10-24T16:55:00.123456Z"],
    )


class ErrorResponse(BaseModel):
    """
    Struktur pembungkus utama untuk seluruh respons galat API.

    Memastikan konsistensi format pesan kesalahan pada tingkat antarmuka klien.
    """

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail = Field(
        ...,
        description="Objek detail informasi kesalahan",
    )


class TimingDetail(BaseModel):
    """
    Rincian durasi waktu eksekusi setiap tahapan inferensi.

    Menyimpan metrik latensi prapemrosesan, inferensi model, dan total durasi proses.
    """

    model_config = ConfigDict(extra="forbid")

    preprocess_ms: float = Field(
        ...,
        description="Waktu tahap pra-pemrosesan citra (milidetik)",
        examples=[3.42],
    )
    inference_ms: float = Field(
        ...,
        description="Waktu eksekusi inferensi model (milidetik)",
        examples=[11.25],
    )
    total_ms: float = Field(
        ...,
        description="Total durasi siklus inferensi (milidetik)",
        examples=[14.67],
    )


class InputImageMetadata(BaseModel):
    """
    Metadata teknis dari citra masukan yang berhasil didekode.

    Memuat dimensi resolusi, jumlah kanal warna, format asli, dan ukuran berkas.
    """

    model_config = ConfigDict(extra="forbid")

    width: int = Field(
        ...,
        description="Lebar gambar asli dalam satuan piksel",
        examples=[1024],
    )
    height: int = Field(
        ...,
        description="Tinggi gambar asli dalam satuan piksel",
        examples=[768],
    )
    channels: int = Field(
        ...,
        description="Jumlah kanal warna gambar (RGB = 3)",
        examples=[3],
    )
    format: str = Field(
        ...,
        description="Format gambar terdeteksi (JPEG atau PNG)",
        examples=["JPEG"],
    )
    file_size_bytes: int = Field(
        ...,
        description="Ukuran berkas dalam satuan byte",
        examples=[142050],
    )


class PredictionResponse(BaseModel):
    """
    Payload hasil prediksi klasifikasi citra cuaca.

    Menyajikan kelas cuaca terpilih, nilai keyakinan softmax, distribusi skor seluruh kelas,
    serta rincian latensi inferensi.
    """

    model_config = ConfigDict(extra="forbid")

    predicted_class: str = Field(
        ...,
        description=(
            "Label kelas cuaca dengan skor tertinggi (cloudy, foggy, rainy, "
            "shine, sunrise)"
        ),
        examples=["shine"],
    )
    class_id: int = Field(
        ...,
        description=(
            "Indeks numerik kelas cuaca (0: cloudy, 1: foggy, 2: rainy, "
            "3: shine, 4: sunrise)"
        ),
        examples=[3],
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Nilai skor softmax untuk kelas hasil prediksi",
        examples=[0.97],
    )
    class_scores: dict[str, float] = Field(
        ...,
        description="Nilai skor keluaran model untuk seluruh 5 kelas cuaca",
        examples=[
            {
                "cloudy": 0.003124,
                "foggy": 0.001892,
                "rainy": 0.004211,
                "shine": 0.985431,
                "sunrise": 0.005342,
            }
        ],
    )
    model_version: str = Field(
        ...,
        description="Versi artefak model yang aktif dimuat",
        examples=["1.0.0"],
    )
    request_id: str = Field(
        ...,
        description="ID unik permintaan untuk penelusuran pada catatan log",
        examples=["c4e5cb8b-2794-4c12-8f4b-fadf3b941ca2"],
    )
    timestamp: str = Field(
        ...,
        description=(
            "Waktu prediksi selesai diproses dalam format UTC ISO 8601"
        ),
        examples=["2026-10-24T16:55:00.123456Z"],
    )
    timing: TimingDetail = Field(
        ...,
        description="Rincian durasi waktu pemrosesan (milidetik)",
    )
    input_metadata: InputImageMetadata = Field(
        ...,
        description="Informasi teknis gambar masukan yang didekode",
    )


class LivenessResponse(BaseModel):
    """
    Status kelangsungan proses aplikasi server HTTP.

    Digunakan oleh orkestrator sistem untuk memastikan kontainer aplikasi
    aktif berjalan.
    """

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ...,
        description="Status operasional proses aplikasi (pass)",
        examples=["pass"],
    )
    service_id: str = Field(
        ...,
        description="Identitas layanan aplikasi",
        examples=["weather-image-classifier"],
    )
    version: str = Field(
        ...,
        description="Versi aplikasi API",
        examples=["1.0.0"],
    )
    timestamp: str = Field(
        ...,
        description="Waktu pemeriksaan dalam format UTC ISO 8601",
        examples=["2026-10-24T16:55:00.123456Z"],
    )
    uptime_seconds: float = Field(
        ...,
        description="Durasi proses server berjalan dalam detik",
        examples=[3612.45],
    )


class ModelCheckDetail(BaseModel):
    """
    Informasi diagnostik kesiapan dan spesifikasi artefak model Keras.

    Menyajikan status ketersediaan, nama model, bentuk tensor masukan,
    dan jumlah kelas keluaran.
    """

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ...,
        description="Status kesiapan artefak model (pass)",
        examples=["pass"],
    )
    model_name: str = Field(
        ...,
        description="Nama arsitektur model",
        examples=["MobileNetV2-WeatherClassifier"],
    )
    model_version: str = Field(
        ...,
        description="Versi artefak model Keras",
        examples=["1.0.0"],
    )
    input_shape: tuple[int | None, ...] = Field(
        ...,
        description="Bentuk dimensi tensor masukan model",
        examples=[(None, 224, 224, 3)],
    )
    num_classes: int = Field(
        ...,
        description="Jumlah kelas keluaran model",
        examples=[5],
    )
    classes: tuple[str, ...] = Field(
        ...,
        description="Daftar nama kelas cuaca keluaran model",
        examples=[("cloudy", "foggy", "rainy", "shine", "sunrise")],
    )
    framework: str = Field(
        ...,
        description="Pustaka komputasi model",
        examples=["Keras 3 / TensorFlow"],
    )
    is_ready: bool = Field(
        ...,
        description="Indikator kesiapan model untuk inferensi",
        examples=[True],
    )


class EngineCheckDetail(BaseModel):
    """
    Informasi diagnostik kesiapan perangkat komputasi inferensi.

    Menampilkan status kesiapan dan tipe perangkat eksekusi model (misalnya CPU).
    """

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ...,
        description="Status kesiapan perangkat eksekusi (pass)",
        examples=["pass"],
    )
    device: str = Field(
        ...,
        description="Perangkat keras yang mengeksekusi inferensi",
        examples=["CPU"],
    )


class ReadinessChecks(BaseModel):
    """
    Kumpulan hasil pemeriksaan kesiapan komponen model dan perangkat komputasi.

    Menggabungkan rincian diagnostik model dan sistem eksekusi inferensi.
    """

    model_config = ConfigDict(extra="forbid")

    model: ModelCheckDetail = Field(
        ...,
        description="Status kesiapan artefak dan kontrak model",
    )
    inference_engine: EngineCheckDetail = Field(
        ...,
        description="Status kesiapan perangkat eksekusi inferensi",
    )


class ReadinessResponse(BaseModel):
    """
    Status kesiapan menyeluruh layanan inferensi untuk menerima beban kerja.

    Menandakan apakah aplikasi dan model siap memproses permintaan prediksi dari klien.
    """

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ...,
        description="Status kesiapan keseluruhan layanan (pass)",
        examples=["pass"],
    )
    service_id: str = Field(
        ...,
        description="Identitas layanan aplikasi",
        examples=["weather-image-classifier"],
    )
    version: str = Field(
        ...,
        description="Versi aplikasi API",
        examples=["1.0.0"],
    )
    timestamp: str = Field(
        ...,
        description="Waktu pemeriksaan dalam format UTC ISO 8601",
        examples=["2026-10-24T16:55:00.123456Z"],
    )
    uptime_seconds: float = Field(
        ...,
        description="Durasi proses server berjalan dalam detik",
        examples=[3612.45],
    )
    checks: ReadinessChecks = Field(
        ...,
        description=(
            "Rincian status pemeriksaan komponen model dan sistem"
        ),
    )


HealthStatus = LivenessResponse
