try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

from . import io, scalars, sim
from .arrays import GenotypeDtype, GenotypeArray
from .accessors import GenotypeSeriesAccessor, GenotypeDataframeAccessor
from pandas_genomics.accessors.utils import generate_weighted_encodings

__version__ = importlib_metadata.version(__name__)

__all__ = [
    __version__,
    GenotypeSeriesAccessor,
    GenotypeDataframeAccessor,
    GenotypeDtype,
    GenotypeArray,
    io,
    scalars,
    sim,
    generate_weighted_encodings,
]
