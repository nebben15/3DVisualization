#!/usr/bin/env python3
"""
Convenience entry point to run the 3D Visualization app.
Usage:
    python run.py

This script works whether you use the project in script mode or as a package.
"""
import sys
import os

# Ensure src/ is on sys.path when running from repo root
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Import app and run
try:
    from src import app  # package-style
except Exception:
    import app  # script-style

if __name__ == "__main__":
    # If app exposes main(), use it; otherwise, run Application
    if hasattr(app, "main"):
        app.main()
    else:
        # Fallback: try to construct and run VisualizationApp
        try:
            from open3d.visualization import gui
            viewer = app.VisualizationApp()
            gui.Application.instance.run()
        except Exception as e:
            print(f"Failed to start app: {e}")
