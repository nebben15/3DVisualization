from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

SUPPORTED_PCD_EXTS = {".ply", ".pcd", ".xyz"}
SUPPORTED_MESH_EXTS = {".obj", ".stl", ".off", ".gltf", ".glb"}
SUPPORTED_EXTS = SUPPORTED_PCD_EXTS | SUPPORTED_MESH_EXTS


@dataclass
class GeometryInfo:
    display: str
    path: str
    type: str  # "pcd" or "mesh"


def scan_directories(source_dirs: List[str]) -> List[GeometryInfo]:
    items: List[GeometryInfo] = []
    for d in source_dirs:
        dp = Path(d)
        if not dp.exists():
            continue
        for p in sorted(dp.glob("**/*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                ext = p.suffix.lower()
                gtype = "pcd" if ext in SUPPORTED_PCD_EXTS else "mesh"
                display = f"{dp.name}/{p.name}"
                items.append(GeometryInfo(display, str(p), gtype))
    return items
