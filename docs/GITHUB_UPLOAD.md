# Publication Checklist

Use this checklist before merging a portfolio refresh into the public default branch.

## Content

- Keep technical claims linked to source, tests or recorded evidence.
- Leave unmeasured success, collision and completion-time fields as `TBD`.
- Confirm that no API key, token, private path, bag file or raw experiment dump is tracked.
- Retain upstream licenses and [source attribution](SOURCE_ATTRIBUTION.md).

## Local Validation

```bash
git status --short
git diff --check
python3 -m pip install pillow
python3 scripts/check_media.py
PYTHONPATH=src python3 -m pytest -q -m "not ros"
```

Run ROS-dependent tests in a sourced ROS 2 Humble workspace:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 -m pytest -q -m ros
```

## Media

- `docs/media/after_training.gif` is the README preview.
- `docs/media/drl_simulation_sequence.mp4` is the H.264/yuv420p browser-facing video.
- Every README media link must be repository-relative.
- Verify the public GitHub page while signed out before claiming that playback works.

## Publish

Review the branch diff and CI results before opening a pull request. Do not force-push the default branch. Record the final public URL and verification date in the external portfolio report.
