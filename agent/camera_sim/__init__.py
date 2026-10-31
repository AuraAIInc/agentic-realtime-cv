"""camera_sim: replay recorded images through the real camera_v3 pipeline.

Streams detection JSON in exactly the camera_server format, without the
view-info API. All constants live in sim_config.yaml.
"""

import os
import sys

# Make `import camera_v3` work regardless of the working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CAMERA_SERVER_DIR = os.path.join(_REPO_ROOT, "camera_server")
if os.path.isdir(os.path.join(_CAMERA_SERVER_DIR, "camera_v3")) and _CAMERA_SERVER_DIR not in sys.path:
    sys.path.insert(0, _CAMERA_SERVER_DIR)

# Keep Ultralytics global state (settings.json, runs/) inside this project.
# Ultralytics unconditionally creates a save dir under its runs_dir on every
# track()/predict() call, and the default runs_dir comes from the user's
# global settings file. Point it here before ultralytics is first imported.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_PACKAGE_DIR, ".ultralytics"))

__all__ = []
