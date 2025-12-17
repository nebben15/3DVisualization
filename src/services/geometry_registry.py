from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

SUPPORTED_PCD_EXTS = {".pcd", ".xyz"}  # .ply handled dynamically
SUPPORTED_MESH_EXTS = {".obj", ".stl", ".off", ".gltf", ".glb"}
SUPPORTED_EXTS = SUPPORTED_PCD_EXTS | SUPPORTED_MESH_EXTS | {".ply"}


@dataclass
class GeometryInfo:
    display: str
    path: str
    type: str  # "pcd" or "mesh"


def scan_directories(source_dirs: List[str]) -> List[GeometryInfo]:
    """
    Recursively crawl all provided directories (depth-unlimited) and collect
    supported geometry files. Display shows relative path from the source dir root
    for clarity when nested folders are present.
    """
    items: List[GeometryInfo] = []
    for d in source_dirs:
        dp = Path(d)
        if not dp.exists():
            continue
        # Use rglob for depth-unlimited recursion
        for p in sorted(dp.rglob("*")):
            if not (p.is_file() and p.suffix.lower() in SUPPORTED_EXTS):
                continue
            ext = p.suffix.lower()
            gtype: str
            if ext == ".ply":
                # Classify PLY by header: if it has faces -> mesh, else -> pcd
                gtype = _classify_ply(str(p))
            elif ext in SUPPORTED_PCD_EXTS:
                gtype = "pcd"
            else:
                gtype = "mesh"
            # Display as "rootdir/relative/subpath/filename"
            try:
                rel = p.relative_to(dp)
                display = f"{dp.name}/{rel.as_posix()}"
            except Exception:
                display = f"{dp.name}/{p.name}"
            items.append(GeometryInfo(display, str(p), gtype))
    return items


def _classify_ply(path: str) -> str:
    """
    Inspect the PLY header to classify as mesh (has faces) vs point cloud.
    Returns "mesh" if an element face is present, otherwise "pcd".
    Works for both ASCII and binary PLY since headers are ASCII.
    """
    try:
        # Read header up to end_header
        with open(path, "rb") as f:
            header_bytes = b""
            while True:
                line = f.readline()
                if not line:
                    break
                header_bytes += line
                if line.strip() == b"end_header":
                    break
        header = header_bytes.decode("utf-8", errors="ignore").lower()
        if "element face" in header:
            return "mesh"
    except Exception:
        pass
    return "pcd"
    return items
