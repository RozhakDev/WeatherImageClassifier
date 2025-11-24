"""
Pengujian unit untuk modul logging terstruktur berbasis JSON.

Memverifikasi serialisasi rekaman log standar dan pemetaan atribut kustom
ke dalam representasi string JSON yang terstruktur.
"""

import json
import logging
import pytest

from src.core.logging import StructuredJsonFormatter


@pytest.mark.unit
class TestStructuredJsonLoggingUnit:
    """
    Rangkaian pengujian unit untuk kelas StructuredJsonFormatter.

    Memastikan setiap rekaman catatan log tersusun rapi dalam format JSON
    dengan menyertakan field standar maupun parameter kontekstual tambahan.
    """

    @pytest.fixture
    def formatter(self) -> StructuredJsonFormatter:
        """
        Menyediakan instance formatter JSON untuk pengujian.

        Returns:
            StructuredJsonFormatter: Objek formatter log berbasis JSON.
        """
        return StructuredJsonFormatter()

    def test_format_serializes_standard_record(
        self,
        formatter: StructuredJsonFormatter,
    ) -> None:
        """
        Memverifikasi bahwa rekaman log standar diserialisasi dengan atribut wajib.

        Args:
            formatter (StructuredJsonFormatter): Objek formatter JSON.
        """
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="module.py",
            lineno=10,
            msg="Proses inferensi selesai.",
            args=(),
            exc_info=None,
            func="infer",
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Proses inferensi selesai."
        assert parsed["source"] == "module.py:10"
        assert parsed["function"] == "infer"
        assert "timestamp" in parsed

    def test_format_unpacks_custom_extra_fields(
        self,
        formatter: StructuredJsonFormatter,
    ) -> None:
        """
        Memverifikasi bahwa atribut tambahan (extra fields) dimasukkan ke dalam JSON keluaran.

        Args:
            formatter (StructuredJsonFormatter): Objek formatter JSON.
        """
        record = logging.LogRecord(
            name="test.metric",
            level=logging.DEBUG,
            pathname="metrics.py",
            lineno=25,
            msg="Latensi metrik dicatat",
            args=(),
            exc_info=None,
            func="record_latency",
        )
        record.request_id = "test-req-999"
        record.duration_ms = 14.85
        record.endpoint = "/v1/predictions"

        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["request_id"] == "test-req-999"
        assert parsed["duration_ms"] == 14.85
        assert parsed["endpoint"] == "/v1/predictions"
