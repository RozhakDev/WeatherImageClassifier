"""
Paket router pengarah endpoint HTTP API.

Mengekspos router pemeriksaan kesehatan layanan dan router inferensi prediksi cuaca.
"""

from .health import health_router
from .predictions import predictions_router

__all__ = ["health_router", "predictions_router"]
