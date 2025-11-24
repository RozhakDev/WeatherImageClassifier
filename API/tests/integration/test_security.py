"""
Pengujian integrasi untuk protokol keamanan dan proteksi serangan HTTP.

Memverifikasi pembatasan ukuran payload, pencegahan MIME spoofing,
penerapan header keamanan HTTP, serta pencegahan kebocoran stack trace internal.
"""

from fastapi.testclient import TestClient
import pytest

from src.config import settings


@pytest.mark.integration
@pytest.mark.security
class TestSecurityProtocols:
    """
    Rangkaian pengujian protokol keamanan sistem inferensi.

    Menguji ketahanan aplikasi terhadap muatan berkas berlebih, manipulasi ekstensi berkas,
    serta memastikan kepatuhan header keamanan jaringan.
    """

    def test_payload_too_large_returns_413(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Memverifikasi bahwa unggahan yang melampaui batas ukuran ditolak dengan HTTP 413.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
            monkeypatch (pytest.MonkeyPatch): Utilitas monkeypatch untuk memodifikasi batas muatan.
        """
        monkeypatch.setattr(settings, "max_upload_bytes", 50)
        oversized_data = b"x" * 200
        response = client.post(
            "/v1/predictions",
            files={"image": ("large.jpg", oversized_data, "image/jpeg")},
        )
        assert response.status_code == 413
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "PAYLOAD_TOO_LARGE"
        assert data["error"]["status"] == 413

    def test_mime_spoofing_rejected(self, client: TestClient) -> None:
        """
        Memverifikasi bahwa berkas teks dengan ekstensi gambar ditolak oleh validasi Pillow.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
        """
        fake_jpeg = b"This is plain text pretending to be a valid jpeg image."
        response = client.post(
            "/v1/predictions",
            files={"image": ("malicious.jpg", fake_jpeg, "image/jpeg")},
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "UNPROCESSABLE_IMAGE"
        assert data["error"]["status"] == 422

    def test_security_headers_present(self, client: TestClient) -> None:
        """
        Memverifikasi kehadiran header keamanan standar pada seluruh respons endpoint.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
        """
        response = client.get("/health/live")
        headers = response.headers
        assert headers.get("X-Content-Type-Options") == settings.security_content_type_options
        assert headers.get("X-Frame-Options") == settings.security_frame_options
        assert headers.get("Strict-Transport-Security") == settings.security_hsts_header
        assert headers.get("Content-Security-Policy") == settings.default_content_security_policy

    def test_docs_security_headers_permit_cdn(self, client: TestClient) -> None:
        """
        Memverifikasi kebijakan CSP pada halaman dokumentasi mengizinkan pemuatan aset CDN.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
        """
        response = client.get("/docs")
        assert response.status_code == 200
        csp = response.headers.get("Content-Security-Policy", "")
        assert "https://cdn.jsdelivr.net" in csp
        assert "'unsafe-inline'" in csp

    def test_request_id_custom_and_auto_generation(self, client: TestClient) -> None:
        """
        Memverifikasi kelangsungan header X-Request-ID baik dari klien maupun pembuatan otomatis.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
        """
        custom_id = "custom-uuid-12345"
        response_custom = client.get(
            "/health/live",
            headers={"X-Request-ID": custom_id},
        )
        assert response_custom.headers.get("X-Request-ID") == custom_id

        response_auto = client.get("/health/live")
        assert "X-Request-ID" in response_auto.headers
        assert len(response_auto.headers["X-Request-ID"]) > 10

    def test_no_stack_trace_leak_in_error_response(self, client: TestClient) -> None:
        """
        Memverifikasi bahwa pesan galat tidak membocorkan stack trace internal kode Python.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
        """
        sample_file = ("error.bin", b"non_image_bytes", "application/octet-stream")
        response = client.post(
            "/v1/predictions",
            files={"image": sample_file},
        )
        assert response.status_code in (415, 422)
        response_text = response.text
        assert "Traceback" not in response_text
        assert 'File "' not in response_text
