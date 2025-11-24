"""
Pengujian integrasi untuk endpoint inferensi klasifikasi citra cuaca.

Memverifikasi pengiriman berkas citra berformat JPEG dan PNG, struktur payload respons,
probabilitas kelas, serta rincian metrik latensi inferensi.
"""

import pytest
from fastapi.testclient import TestClient

from src.domain.constants import WEATHER_CLASSES


@pytest.mark.integration
class TestPredictionsEndpoint:
    """
    Rangkaian pengujian integrasi untuk endpoint /v1/predictions.

    Menguji respons inferensi secara end-to-end dengan payload gambar nyata
    pada lingkungan aplikasi FastAPI yang berjalan penuh.
    """

    @pytest.mark.parametrize(
        "fixture_name,filename,mime_type,expected_format",
        [
            ("valid_jpeg_bytes", "sample.jpg", "image/jpeg", "JPEG"),
            ("valid_png_bytes", "sample.png", "image/png", "PNG"),
        ],
    )
    def test_prediction_with_supported_image_formats(
        self,
        client: TestClient,
        request: pytest.FixtureRequest,
        fixture_name: str,
        filename: str,
        mime_type: str,
        expected_format: str,
    ) -> None:
        """
        Memverifikasi bahwa pengunggahan citra JPEG dan PNG menghasilkan respons prediksi valid.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
            request (pytest.FixtureRequest): Permintaan fixture pytest dinamis.
            fixture_name (str): Nama fixture penyedia data byte citra.
            filename (str): Nama berkas citra unggahan.
            mime_type (str): Tipe MIME header pengunggahan berkas.
            expected_format (str): Format citra yang diharapkan pada metadata.
        """
        image_bytes: bytes = request.getfixturevalue(fixture_name)
        response = client.post(
            "/v1/predictions",
            files={"image": (filename, image_bytes, mime_type)},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["predicted_class"] in WEATHER_CLASSES
        assert isinstance(data["class_id"], int)
        assert 0 <= data["class_id"] < len(WEATHER_CLASSES)
        assert 0.0 <= data["score"] <= 1.0
        assert len(data["class_scores"]) == 5
        assert set(data["class_scores"].keys()) == set(WEATHER_CLASSES)
        assert data["predicted_class"] == max(
            data["class_scores"], key=data["class_scores"].get
        )
        assert data["score"] == data["class_scores"][data["predicted_class"]]
        assert pytest.approx(sum(data["class_scores"].values()), abs=0.01) == 1.0
        assert data["model_version"] == "1.0.0"

        assert "timing" in data
        assert data["timing"]["preprocess_ms"] >= 0.0
        assert data["timing"]["inference_ms"] >= 0.0
        assert data["timing"]["total_ms"] >= 0.0

        assert "input_metadata" in data
        assert data["input_metadata"]["format"] == expected_format
        assert data["input_metadata"]["channels"] == 3
        assert data["input_metadata"]["file_size_bytes"] == len(image_bytes)

        assert "request_id" in data
        assert "timestamp" in data
