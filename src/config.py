from pathlib import Path

# Default directories to scrape for geometries (meshes/point clouds).
# These are safe guesses; non-existent paths are ignored at runtime.
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEOMETRY_DIRS = [
	str(_ROOT / "samples"),                # e.g., ../samples
	str(_ROOT / "data" / "shapes"),      # e.g., ../data/shapes
]

# Optionally include a user-specific path if present. Commented out by default.
from pathlib import Path as _P
DEFAULT_GEOMETRY_DIRS.append(str(_P.home() / "LRZ Sync+Share/Thesis/Data/samples"))
DEFAULT_GEOMETRY_DIRS.append(str(_P.home() / "LRZ Sync+Share/Thesis/Data/data/shapes"))

