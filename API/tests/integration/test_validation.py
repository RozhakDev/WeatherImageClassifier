"""
Pengujian integrasi untuk validasi permintaan dan pemetaan kode galat HTTP.

Memverifikasi penolakan field form tidak valid, berkas kosong, format tidak didukung,
citra rusak, serta kondisi ketika model inferensi belum siap.
"""

from typing import Any
from fastapi.testclient import TestClient
import pytest


@pytest.mark.integration
@pytest.mark.validation
class TestRequestValidation:
    """
    Rangkaian pengujian integrasi untuk validasi request payload.

    Memastikan setiap bentuk galat masukan menghasilkan kode status HTTP
    dan objek error yang konsisten sesuai standar RFC 7807 / 9457.
    """

    @pytest.mark.parametrize(
        "files_payload,expected_status,expected_code",
        [
            (
                {"document": ("sample.jpg", b"dummy_data", "image/jpeg")},
                400,
                "INVALID_REQUEST",
            ),
            (
                {"image": ("empty.jpg", b"", "image/jpeg")},
                400,
                "INVALID_REQUEST",
            ),
        ],
    )
    def test_invalid_request_payload_returns_400(
        self,
        client: TestClient,
        files_payload: dict[str, Any],
        expected_status: int,
        expected_code: str,
    ) -> None:
        """
        Memverifikasi bahwa field form tidak lengkap atau berkas kosong memicu respons HTTP 400.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
            files_payload (dict[str, Any]): Payload berkas form-data.
            expected_status (int): Kode status HTTP yang diharapkan.
            expected_code (str): Kode string kesalahan yang diharapkan.
        """
        response = client.post("/v1/predictions", files=files_payload)
        assert response.status_code == expected_status
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == expected_code
        assert data["error"]["status"] == expected_status

    def test_unsupported_format_returns_415(
        self,
        client: TestClient,
        valid_gif_bytes: bytes,
    ) -> None:
        """
        Memverifikasi bahwa format citra tidak didukung memicu respons HTTP 415.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
            valid_gif_bytes (bytes): Konten biner citra GIF.
        """
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.gif", valid_gif_bytes, "image/gif")},
        )
        assert response.status_code == 415
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"
        assert data["error"]["status"] == 415

    def test_corrupt_image_returns_422(
        self,
        client: TestClient,
        corrupt_bytes: bytes,
    ) -> None:
        """
        Memverifikasi bahwa berkas biner rusak memicu respons HTTP 422.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
            corrupt_bytes (bytes): Data biner rusak.
        """
        response = client.post(
            "/v1/predictions",
            files={"image": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "UNPROCESSABLE_IMAGE"
        assert data["error"]["status"] == 422

    def test_model_not_ready_returns_503(
        self,
        client: TestClient,
        valid_jpeg_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Memverifikasi bahwa permintaan prediksi mengembalikan HTTP 503 saat model belum siap.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
            valid_jpeg_bytes (bytes): Data biner citra JPEG valid.
            monkeypatch (pytest.MonkeyPatch): Utilitas monkeypatch pytest.
        """
        monkeypatch.setattr(client.app.state.model_service, "_is_ready", False)
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.jpg", valid_jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "MODEL_UNAVAILABLE"
        assert data["error"]["status"] == 503
