# WeatherImageClassifier

Layanan REST API inferensi untuk klasifikasi citra cuaca ke dalam 5 kelas (`cloudy`, `foggy`, `rainy`, `shine`, `sunrise`) menggunakan model MobileNetV2 dan FastAPI.

## Arsitektur

Arsitektur sistem memisahkan pipeline pelatihan model (offline) dan layanan inferensi REST API (online serving):

- **Separation of Concerns**: Memisahkan entitas domain (`schemas.py`, `constants.py`), logika pemrosesan dan inferensi (`services/`), routing HTTP (`routers/`), serta konfigurasi infrastruktur (`core/`).
- **Model Lifecycle**: Model Keras dimuat satu kali saat startup aplikasi melalui FastAPI `lifespan` handler untuk meminimalkan latensi inferensi dan mencegah pemuatan ulang berulang.
- **In-Memory Processing**: Berkas citra didekode dan divalidasi langsung di memori RAM menggunakan Pillow tanpa penyimpanan ke disk server.

## Machine Learning

### Dataset

- **Sumber Data**: Dataset publik Kaggle [Multiclass Weather Dataset](https://www.kaggle.com/datasets/vijaygiitk/multiclass-weather-dataset) (`vijaygiitk/multiclass-weather-dataset`).

- **Distribusi Kelas**: Total 1.500 citra yang terbagi ke dalam 5 kelas cuaca:
  
  | Kelas     | Train (60%) | Val (20%) | Test (20%) | Total     |
  |:--------- |:-----------:|:---------:|:----------:|:---------:|
  | `cloudy`  | 180         | 60        | 60         | 300       |
  | `foggy`   | 180         | 60        | 60         | 300       |
  | `rainy`   | 180         | 60        | 60         | 300       |
  | `shine`   | 150         | 50        | 50         | 250       |
  | `sunrise` | 210         | 70        | 70         | 350       |
  | **Total** | **900**     | **300**   | **300**    | **1.500** |

- **Prapemrosesan**: Resolusi target `224 × 224` piksel RGB dengan normalisasi `mobilenet_v2.preprocess_input`. Augmentasi (rotasi 20°, zoom 20%, shear 20%, shift 20%, horizontal flip) hanya diterapkan pada data latih.

### Notebook

Eksperimen model diimplementasikan pada [`DeepLearning/notebook/WeatherImageClassifier.ipynb`](DeepLearning/notebook/WeatherImageClassifier.ipynb). Tahapan mencakup eksplorasi data, pembagian dataset terstratifikasi (seed 42), konfigurasi generator data, perakitan arsitektur transfer learning, pelatihan dua tahap, evaluasi data uji, dan ekspor artefak Keras.

### Model

- **Backbone**: MobileNetV2 dengan bobot pralatih ImageNet (`include_top=False`, `input_shape=(224, 224, 3)`).
- **Classification Head**: `Conv2D(32, (3, 3), relu, same)` -> `MaxPooling2D((2, 2))` -> `GlobalAveragePooling2D()` -> `Dense(1024, relu)` -> `BatchNormalization()` -> `Dropout(0.5)` -> `Dense(5, softmax)`.
- **Artefak**: `models/weather_classifier.keras` dengan dimensi masukan `(None, 224, 224, 3)` dan keluaran `(None, 5)`.
- **Pemetaan Indeks Kelas**: `0: cloudy`, `1: foggy`, `2: rainy`, `3: shine`, `4: sunrise`.

### Pelatihan dan Evaluasi

- **Tahap Pelatihan**:
  - *Tahap 1 (Head Training)*: Backbone dibekukan, Adam `lr=1e-3`, `categorical_crossentropy`, 20 epoch. Callbacks: `ReduceLROnPlateau`, `EarlyStopping`, dan `ModelCheckpoint` (`best_model_stage1.keras` memantau `val_loss`).
  - *Tahap 2 (Fine-Tuning)*: Membuka lapisan backbone mulai indeks 100, Adam `lr=1e-5`, batas total 30 epoch. Callbacks: `EarlyStopping`, `ModelCheckpoint` (`best_model_fine_tuned.keras` memantau `val_accuracy`), dan `ReduceLROnPlateau`.
- **Hasil Evaluasi**:
  - Data Latih (900 citra): Loss `0.1488` | Akurasi **95.11%**
  - Data Uji (300 citra): Loss `0.3092` | Akurasi **92.00%**

## API

### Tanggung Jawab

- Membaca aliran byte berkas unggahan secara bertahap (chunk 64 KB) dan membatasi ukuran muatan (`MAX_UPLOAD_BYTES = 10 MB`).
- Memvalidasi isi biner berkas gambar (JPEG dan PNG) menggunakan Pillow untuk mencegah MIME spoofing.
- Mengonversi citra ke RGB 224x224, menerapkan transformasi tensor MobileNetV2, dan mengeksekusi inferensi pada model Keras.
- Menyusun respons terstruktur yang memuat kelas prediksi, skor keyakinan, distribusi skor seluruh kelas, metrik latensi inferensi, dan metadata citra.
- Menyediakan probe diagnostik liveness dan readiness serta respons galat terstandarisasi RFC 7807/9457.
- *Batasan Sistem*: Layanan ini tidak menyediakan autentikasi pengguna, penyimpanan berkas persisten, atau inferensi batch asinkron.

### Struktur Implementasi

```text
API/src/
├── services/   # Prapemrosesan citra & inferensi model
├── routers/    # Handler endpoint & dependensi
├── core/       # Middleware, logging, & penanganan error
├── domain/     # Skema data Pydantic & konstanta
├── config.py   # Konfigurasi aplikasi
└── main.py     # Titik masuk FastAPI & lifespan
```

### Instalasi dan Menjalankan API

1. **Persiapan Virtual Environment**:
   
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\Activate.ps1 | Linux/macOS: source .venv/bin/activate
   pip install -r API/requirements.txt
   ```
2. **Konfigurasi Lingkungan**:
   Salin `API/.env.example` ke `API/.env` dan sesuaikan parameter (port 8000, model path `models/weather_classifier.keras`, batas upload 10 MB).
3. **Menjalankan Server**:
   
   ```bash
   cd API
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Endpoint dan Skema Respons

Dokumentasi skema permintaan dan respons interaktif tersedia melalui Swagger UI (`/docs`), ReDoc (`/redoc`), dan spesifikasi OpenAPI (`/openapi.json`).


| Metode | Jalur Endpoint | Ringkasan Fungsi | Status Sukses | Tipe Konten / Parameter |
| :---: | :--- | :--- | :---: | :--- |
| **`POST`** | `/v1/predictions` | Klasifikasi citra cuaca (cloudy, foggy, rainy, shine, sunrise) | `200 OK` | `multipart/form-data` (`image`: JPEG/PNG, maks 10 MB) |
| **`GET`** | `/health/live` | Pemeriksaan proses aplikasi server HTTP | `200 OK` | `application/json` (status pass, service_id, uptime) |
| **`GET`** | `/health/ready` | Pemeriksaan kesiapan model Keras dan engine inferensi | `200 OK` | `application/json` (status pass, model check, device) |

Respons galat terstandarisasi mengadopsi format RFC 7807 / RFC 9457 untuk kode status `400`, `413`, `415`, `422`, `500`, dan `503`.

## Pengujian

Rangkaian pengujian otomatis dibangun menggunakan Pytest dengan total 40 skenario pengujian (100% lulus):

- **Unit Testing (`API/tests/unit/`, 20 skenario)**: Pengujian terisolasi murni dalam memori tanpa menyalakan server HTTP (~0.5 detik). Menguji konfigurasi (`test_config.py`), pemrosesan Pillow (`test_image_processor.py`), serialisasi log JSON (`test_logging.py`), dan alur orkestrasi prediksi (`test_prediction_service.py`).
- **Integration Testing (`API/tests/integration/`, 20 skenario)**: Pengujian antarmuka HTTP berbasis `TestClient`. Menguji spesifikasi OpenAPI (`test_docs.py`), probe kesehatan (`test_health.py`), inferensi berkas JPEG dan PNG (`test_predictions.py`), protokol keamanan HTTP (`test_security.py`), dan penanganan input galat (`test_validation.py`).
- **Perintah Eksekusi**:
  
  ```bash
  python -m pytest                # Menjalankan seluruh 40 pengujian
  python -m pytest -m unit        # Menjalankan unit tests saja
  python -m pytest -m integration # Menjalankan integration tests saja
  ```


## Dependensi

| Kategori | Paket Utama | Fungsi Utama |
| :--- | :--- | :--- |
| **API Runtime** | `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings` | REST API framework, server ASGI, validasi data, dan manajemen konfigurasi `.env`. |
| **Machine Learning** | `tensorflow`, `keras`, `numpy`, `pillow` | Eksekusi model MobileNetV2, komputasi tensor, dan decoding/normalisasi citra. |
| **Eksperimen ML** | `split-folders`, `kaggle`, `matplotlib`, `seaborn` | Pembagian dataset terstratifikasi, pengunduhan data Kaggle, dan visualisasi metrik. |
| **Pengujian** | `pytest`, `pytest-cov`, `httpx` | Pengujian unit modular, pengujian integrasi HTTP, dan pelaporan cakupan kode. |

## Lisensi

Proyek ini dilisensikan di bawah ketentuan [MIT License](LICENSE). Hak Cipta (c) 2025 **Rozhak**.
