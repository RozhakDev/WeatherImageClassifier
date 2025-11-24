"""
Pencatatan log terstruktur berbasis JSON untuk observability aplikasi.

Modul ini menyediakan formatter JSON kustom dan fungsi inisialisasi logging
agar seluruh catatan peristiwa mudah dibaca dan dianalisis oleh sistem monitoring.
"""

from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any

from ..config import settings

BUILTIN_RECORD_ATTRS: frozenset[str] = frozenset(
    (
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "id",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    )
)


class StructuredJsonFormatter(logging.Formatter):
    """
    Format pencatatan log ke dalam representasi string JSON yang rapi.

    Menyusun atribut dasar rekaman log serta mengekstrak variabel kontekstual tambahan
    (extra fields) ke dalam satu objek JSON utuh.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Mengubah rekaman log standar menjadi baris string JSON.

        Args:
            record (logging.LogRecord): Objek rekaman log yang akan diformat.

        Returns:
            str: Representasi JSON berformat string dari rekaman log.
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "source": f"{record.filename}:{record.lineno}",
            "function": record.funcName,
        }

        for key, value in record.__dict__.items():
            if key not in BUILTIN_RECORD_ATTRS and not key.startswith("_"):
                log_data[key] = value

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


def setup_logging() -> None:
    """
    Mengonfigurasi logger utama aplikasi dengan formatter JSON terstruktur.

    Fungsi ini menyetel level log, membersihkan handler bawaan, serta
    mengarahkan output catatan log ke stdout.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())

    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredJsonFormatter())
    root_logger.addHandler(console_handler)

    logging.getLogger("uvicorn.access").handlers = []
