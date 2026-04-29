from __future__ import annotations

from typing import Any


class DataFile:
    """
    A class representing a DataFile to be made by pipeline stages
    and passed on to subsequent ones.

    DataFile itself should not be instantiated - instead subclasses
    should be defined for different file types.

    These subclasses are used in the definition of pipeline stages
    to indicate what kind of file is expected.  The "suffix" attribute,
    which must be defined on subclasses, indicates the file suffix.

    The open method, which can optionally be overridden, is used by the
    machinery of the PipelineStage class to open an input our output
    named by a tag.

    """

    @classmethod
    def open(cls, path: str, mode: str) -> Any:
        """
        Open a data file.  The base implementation of this function just
        opens and returns a standard python file object.

        Subclasses can override to either open files using different openers
        (like fitsio.FITS), or, for more specific data types, return an
        instance of the class itself to use as an intermediary for the file.

        Parameters
        ----------
        path : str
            Filesystem path to the file.
        mode : str
            File open mode (e.g. ``'r'``, ``'w'``).

        Returns
        -------
        Any
            An open file handle whose type depends on the subclass.
        """
        return open(path, mode)


class HDFFile(DataFile):
    """
    A data file in the HDF5 format.
    Using these files requires the h5py package, which in turn
    requires an HDF5 library installation.

    """

    suffix = "hdf"

    @classmethod
    def open(cls, path: str, mode: str, **kwargs: Any) -> Any:
        """Open an HDF5 file via *h5py*.

        Parameters
        ----------
        path : str
            Filesystem path to the HDF5 file.
        mode : str
            File open mode (e.g. ``'r'``, ``'w'``).
        **kwargs : Any
            Extra keyword arguments forwarded to ``h5py.File``.

        Returns
        -------
        h5py.File
            The opened HDF5 file handle.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import h5py
        return h5py.File(path, mode, **kwargs)


class FitsFile(DataFile):
    """
    A data file in the FITS format.
    Using these files requires the fitsio package.
    """

    suffix = "fits"

    @classmethod
    def open(cls, path: str, mode: str, **kwargs: Any) -> Any:
        """Open a FITS file via *fitsio*.

        Parameters
        ----------
        path : str
            Filesystem path to the FITS file.
        mode : str
            File open mode. ``'w'`` is automatically converted to ``'rw'``
            because fitsio does not support a pure write mode.
        **kwargs : Any
            Extra keyword arguments forwarded to ``fitsio.FITS``.

        Returns
        -------
        fitsio.FITS
            The opened FITS file handle.
        """
        import fitsio

        # Fitsio doesn't have pure 'w' modes, just 'rw'.
        # Maybe we should check if the file already exists here?
        if mode == "w":
            mode = "rw"
        return fitsio.FITS(path, mode=mode, **kwargs)


class TextFile(DataFile):
    """
    A data file in plain text format.
    """

    suffix = "txt"


class YamlFile(DataFile):
    """
    A data file in yaml format.
    """

    suffix = "yml"


class NpzFile(DataFile):
    """A data file in NumPy compressed (``.npz``) format."""

    suffix = "npz"


class DirFile(DataFile):
    """A pseudo-file type representing an output directory."""

    suffix = "dir"


class HTMLFile(DataFile):
    """A data file in HTML format."""

    suffix = "html"


class DummyFile(DataFile):
    """
    A dummy type
    """

    suffix = "dum"

    @classmethod
    def open(cls, path: str, mode: str, **kwargs: Any) -> Any:
        """Open is not supported for dummy files.

        Raises
        ------
        NotImplementedError
            Always raised.
        """
        raise NotImplementedError("Not implemented yet!")
