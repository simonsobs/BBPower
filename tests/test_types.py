"""Tests for bbpower.types — DataFile hierarchy."""
from __future__ import annotations

import pytest

from bbpower.types import (
    DataFile,
    DirFile,
    DummyFile,
    FitsFile,
    HDFFile,
    HTMLFile,
    NpzFile,
    TextFile,
    YamlFile,
)


class TestDataFileOpen:
    """Test the base DataFile.open classmethod."""

    def test_opens_text_file(self, tmp_path: object) -> None:
        """DataFile.open returns a standard file object."""
        p = tmp_path / "hello.txt"
        p.write_text("data")
        fh = DataFile.open(str(p), "r")
        assert fh.read() == "data"
        fh.close()


class TestDummyFileOpen:
    """Test that DummyFile.open raises NotImplementedError."""

    def test_raises(self) -> None:
        """DummyFile.open always raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            DummyFile.open("any_path", "r")


class TestSuffixAttributes:
    """Verify suffix class attributes on all DataFile subclasses."""

    @pytest.mark.parametrize(
        "cls, expected",
        [
            (HDFFile, "hdf"),
            (FitsFile, "fits"),
            (TextFile, "txt"),
            (YamlFile, "yml"),
            (NpzFile, "npz"),
            (DirFile, "dir"),
            (HTMLFile, "html"),
            (DummyFile, "dum"),
        ],
    )
    def test_suffix(self, cls: type, expected: str) -> None:
        """Each subclass defines the correct suffix."""
        assert cls.suffix == expected


class TestClassHierarchy:
    """All concrete file types inherit from DataFile."""

    @pytest.mark.parametrize(
        "cls",
        [HDFFile, FitsFile, TextFile, YamlFile, NpzFile, DirFile, HTMLFile, DummyFile],
    )
    def test_is_subclass(self, cls: type) -> None:
        """Verify subclass relationship."""
        assert issubclass(cls, DataFile)
