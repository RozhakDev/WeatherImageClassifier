"""
Komponen middleware untuk penanganan lalu lintas HTTP.

Modul ini menyediakan middleware penelusuran request ID, penegakan header
keamanan standar, serta pencatatan log akses HTTP.
"""

import logging
import time
from typing import Callable
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import settings

logger = logging.getLogger("src.access")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Menyisipkan atau meneruskan identifier unik untuk setiap permintaan HTTP.

    Memastikan setiap request memiliki nilai header X-Request-ID untuk memudahkan
    penelusuran log di seluruh alur sistem.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Memproses permintaan dan menyematkan identifier unik pada header respons.

        Args:
            request (Request): Objek permintaan HTTP dari klien.
            call_next (Callable): Fungsi handler untuk melanjutkan alur request.

        Returns:
            Response: Objek respons HTTP dengan header X-Request-ID.
        """
        incoming_id = request.headers.get("X-Request-ID")
        request_id = incoming_id if incoming_id else str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Menyematkan header perlindungan keamanan pada setiap respons HTTP.

    Melindungi antarmuka layanan dari celah keamanan umum seperti MIME-sniffing,
    clickjacking, serta serangan injeksi skrip.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Menambahkan konfigurasi header keamanan standar pada objek respons.

        Args:
            request (Request): Objek permintaan HTTP dari klien.
            call_next (Callable): Fungsi handler untuk melanjutkan alur request.

        Returns:
            Response: Objek respons HTTP dengan header keamanan aktif.
        """
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = settings.security_content_type_options
        response.headers["X-Frame-Options"] = settings.security_frame_options
        response.headers["Strict-Transport-Security"] = settings.security_hsts_header

        docs_paths = tuple(path for path in (settings.docs_url, settings.redoc_url, settings.openapi_url) if path)
        if docs_paths and request.url.path.startswith(docs_paths):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: https://fastapi.tiangolo.com;"
            )
        else:
            response.headers["Content-Security-Policy"] = settings.default_content_security_policy
        return response


class AccessLoggingMiddleware(BaseHTTPMiddleware):
    """
    Mencatat metrik waktu dan status penyelesaian setiap permintaan HTTP.

    Membantu observabilitas sistem dengan mengukur durasi siklus request
    serta mencatat galat yang tidak tertangani.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Mengukur durasi eksekusi request dan mencatatnya ke sistem log.

        Args:
            request (Request): Objek permintaan HTTP dari klien.
            call_next (Callable): Fungsi handler untuk melanjutkan alur request.

        Returns:
            Response: Objek respons HTTP dari proses berikutnya.

        Raises:
            Exception: Meneruskan kembali exception yang terjadi selama pemrosesan.
        """
        start_time = time.perf_counter()
        request_id = getattr(request.state, "request_id", "unknown")

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                f"{request.method} {request.url.path} responded {response.status_code} in {duration_ms}ms",
                extra={
                    "request_id": request_id,
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "model_version": settings.model_version,
                },
            )
            return response
        except Exception as error:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                f"{request.method} {request.url.path} failed with unhandled exception: {error}",
                extra={
                    "request_id": request_id,
                    "endpoint": request.url.path,
                    "status_code": 500,
                    "duration_ms": duration_ms,
                    "model_version": settings.model_version,
                },
                exc_info=True,
            )
            raise
