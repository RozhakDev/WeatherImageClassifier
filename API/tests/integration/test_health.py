"""
Pengujian integrasi untuk probe kesehatan liveness dan readiness.

Memverifikasi endpoint /health/live dan /health/ready, termasuk status
operasional sistem serta respons penolakan saat model belum siap.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestHealthEndpoints:
    """
    Rangkaian pengujian integrasi untuk endpoint kesehatan layanan.

    Memastikan probe sistem merespons sesuai standar observability sistem
    termasuk pelaporan uptime dan spesifikasi kontrak model.
    """

    def test_liveness_returns_200(self, client: TestClient) -> None:
        """
        Memverifikasi bahwa probe liveness mengembalikan HTTP 200 dan status pass.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
        """
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pass"
        assert data["service_id"] == "weather-image-classifier"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
        assert data["uptime_seconds"] >= 0.0

    def test_readiness_returns_200_when_ready(self, client: TestClient) -> None:
        """
        Memverifikasi bahwa probe readiness mengembalikan HTTP 200 saat model telah siap.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
        """
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pass"
        assert data["service_id"] == "weather-image-classifier"
        assert "checks" in data

        model_check = data["checks"]["model"]
        assert model_check["status"] == "pass"
        assert model_check["model_name"] == "MobileNetV2-WeatherClassifier"
        assert model_check["input_shape"] == [None, 224, 224, 3]
        assert model_check["num_classes"] == 5
        assert len(model_check["classes"]) == 5
        assert model_check["is_ready"] is True
        assert data["checks"]["inference_engine"]["status"] == "pass"
        assert data["checks"]["inference_engine"]["device"] == "CPU"

    def test_readiness_returns_503_when_not_ready(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Memverifikasi bahwa probe readiness mengembalikan HTTP 503 jika model belum siap.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
            monkeypatch (pytest.MonkeyPatch): Utilitas monkeypatch untuk memodifikasi state model.
        """
        monkeypatch.setattr(client.app.state.model_service, "_is_ready", False)
        response = client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "MODEL_UNAVAILABLE"
        assert data["error"]["status"] == 503
        assert "request_id" in data["error"]
        assert "timestamp" in data["error"]
