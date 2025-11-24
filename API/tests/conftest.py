"""
Konfigurasi fixture pengujian bersama untuk pytest.

Modul ini menyediakan fixture klien HTTP dan payload citra uji
yang digunakan bersama oleh rangkaian pengujian unit dan integrasi.
"""

import io
from typing import Generator
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from src.main import app


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """
    Menyediakan instance TestClient FastAPI untuk seluruh sesi pengujian.

    Returns:
        Generator[TestClient, None, None]: Klien HTTP untuk pengujian integrasi.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def valid_jpeg_bytes() -> bytes:
    """
    Menghasilkan data byte gambar JPEG valid dengan ukuran 224x224 piksel.

    Returns:
        bytes: Konten byte citra berformat JPEG.
    """
    buffer = io.BytesIO()
    image = Image.new("RGB", (224, 224), color=(100, 150, 200))
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def valid_png_bytes() -> bytes:
    """
    Menghasilkan data byte gambar PNG valid dengan ukuran 224x224 piksel.

    Returns:
        bytes: Konten byte citra berformat PNG.
    """
    buffer = io.BytesIO()
    image = Image.new("RGBA", (224, 224), color=(50, 120, 180, 255))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def corrupt_bytes() -> bytes:
    """
    Menyediakan rangkaian byte acak yang tidak dapat didekode sebagai citra.

    Returns:
        bytes: Data biner rusak untuk pengujian validasi gambar.
    """
    return b"CorruptImageDataBytesHere123456789"


@pytest.fixture(scope="session")
def valid_gif_bytes() -> bytes:
    """
    Menghasilkan data byte gambar GIF yang tidak termasuk format yang diizinkan.

    Returns:
        bytes: Konten byte citra berformat GIF.
    """
    buffer = io.BytesIO()
    image = Image.new("RGB", (224, 224), color=(255, 0, 0))
    image.save(buffer, format="GIF")
    return buffer.getvalue()
