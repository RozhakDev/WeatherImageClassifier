"""
Pengujian unit untuk modul konfigurasi aplikasi.

Memverifikasi kemampuan parsing resolusi citra target, format berkas,
dan pencarian lokasi berkas model pada class Settings.
"""

from pathlib import Path
import pytest

from src.config import Settings


@pytest.mark.unit
class TestSettingsConfig:
    """
    Rangkaian pengujian unit untuk validasi konfigurasi Settings.

    Menguji parsing berbagai representasi format string dan struktur data
    ke dalam tipe data konfigurasi yang diharapkan.
    """

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("(224, 224)", (224, 224)),
            ("[224, 224]", (224, 224)),
            ("224, 224", (224, 224)),
            ([224, 224], (224, 224)),
            ((224, 224), (224, 224)),
            ("invalid_shape", (224, 224)),
        ],
    )
    def test_target_image_size_parsing(
        self,
        raw_input: object,
        expected: tuple[int, int],
    ) -> None:
        """
        Memastikan validator resolusi citra mampu memproses teks, list, dan nilai rusak.

        Args:
            raw_input (object): Nilai masukan pengujian.
            expected (tuple[int, int]): Pasangan dimensi yang diharapkan.
        """
        result = Settings.validate_target_image_size(raw_input)
        assert result == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ('["JPEG", "PNG"]', ("JPEG", "PNG")),
            ("JPEG, PNG", ("JPEG", "PNG")),
            (["JPEG", "PNG"], ("JPEG", "PNG")),
            (("JPEG", "PNG"), ("JPEG", "PNG")),
            (12345, ("JPEG", "PNG")),
        ],
    )
    def test_allowed_formats_parsing(
        self,
        raw_input: object,
        expected: tuple[str, ...],
    ) -> None:
        """
        Memastikan validator format ekstensi mampu mengonversi masukan ke tuple string.

        Args:
            raw_input (object): Nilai mentah daftar format.
            expected (tuple[str, ...]): Koleksi tuple format yang diharapkan.
        """
        result = Settings.validate_allowed_formats(raw_input)
        assert result == expected

    def test_resolve_model_path_returns_path_instance(self) -> None:
        """
        Memastikan metode pencarian lokasi berkas model mengembalikan instance Path yang valid.
        """
        settings = Settings()
        resolved = settings.resolve_model_path()
        assert isinstance(resolved, Path)
