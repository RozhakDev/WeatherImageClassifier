"""
Pengujian integrasi untuk kontrak dokumentasi OpenAPI dan Swagger.

Memverifikasi integritas metadata OpenAPI, pengelompokan tag,
operasi endpoint, serta deskripsi atribut pada skema respons.
"""

from fastapi.testclient import TestClient
import pytest


@pytest.mark.integration
class TestDocumentationContract:
    """
    Rangkaian pengujian integrasi spesifikasi OpenAPI.

    Memastikan spesifikasi OpenAPI yang dihasilkan oleh FastAPI
    mematuhi standar dokumentasi API untuk sistem inferensi cuaca.
    """

    def test_openapi_json_metadata(self, client: TestClient) -> None:
        """
        Memverifikasi kelengkapan metadata info, lisensi, dan kontak pada OpenAPI JSON.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
        """
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()

        assert spec["openapi"].startswith("3.")
        info = spec["info"]
        assert info["title"] == "WeatherImageClassifier API"
        assert info["version"] == "1.0.0"
        assert "MobileNetV2" in info["summary"]
        assert "Ruang Lingkup Layanan" in info["description"]
        assert "cloudy" in info["description"]

        assert info["contact"]["name"] == "Tim Pengembang WeatherImageClassifier"
        assert "email" in info["contact"]
        assert info["license"]["name"] == "Apache-2.0"

    def test_openapi_tags_structure(self, client: TestClient) -> None:
        """
        Memverifikasi ketersediaan dan penamaan tag pengelompokan endpoint.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
        """
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()

        tag_names = [t["name"] for t in spec.get("tags", [])]
        assert "Prediksi Cuaca" in tag_names
        assert "Status Layanan" in tag_names

    def test_openapi_endpoint_operations(self, client: TestClient) -> None:
        """
        Memverifikasi operasi HTTP path, summary, dan pemetaan response codes.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
        """
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        paths = spec["paths"]

        assert "/health/live" in paths
        assert "/health/ready" in paths
        assert "/v1/predictions" in paths

        live_op = paths.get("/health/live", {}).get("get") or paths.get("health/live", {}).get("get")
        assert live_op["summary"] == "Pemeriksaan proses aplikasi"
        assert "Status Layanan" in live_op["tags"]

        ready_op = paths["/health/ready"]["get"]
        assert ready_op["summary"] == "Pemeriksaan kesiapan model"
        assert "Status Layanan" in ready_op["tags"]
        assert "503" in ready_op["responses"]

        pred_op = paths["/v1/predictions"]["post"]
        assert pred_op["summary"] == "Klasifikasi citra cuaca"
        assert "Prediksi Cuaca" in pred_op["tags"]
        assert "413" in pred_op["responses"]
        assert "415" in pred_op["responses"]
        assert "422" in pred_op["responses"]
        assert "503" in pred_op["responses"]

    def test_schema_field_descriptions(self, client: TestClient) -> None:
        """
        Memverifikasi bahwa seluruh field pada skema memiliki atribut deskripsi yang jelas.

        Args:
            client (TestClient): Klien pengujian HTTP FastAPI.
        """
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        schemas = spec["components"]["schemas"]

        assert "PredictionResponse" in schemas
        pred_props = schemas["PredictionResponse"]["properties"]
        assert "description" in pred_props["predicted_class"]
        assert "description" in pred_props["score"]
        assert "description" in pred_props["class_scores"]
        assert "description" in pred_props["timing"]
        assert "description" in pred_props["input_metadata"]

        assert "TimingDetail" in schemas
        timing_props = schemas["TimingDetail"]["properties"]
        assert "description" in timing_props["preprocess_ms"]
        assert "description" in timing_props["inference_ms"]
        assert "description" in timing_props["total_ms"]
