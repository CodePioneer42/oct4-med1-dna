"""Map original one-based frame numbers to full or selected-frame DCDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


FRAME_MAP_NAME = "trajectory_frames.json"
FRAME_MAP_SCHEMA = "oct4-med1-selected-frames-v1"


def trajectory_frame_indices(
    directory: Path, frame_count: int, requested_frames: Iterable[int]
) -> tuple[int, ...]:
    """Return storage indices while keeping source frame IDs in output tables.

    Full trajectories need no sidecar. Selected-frame releases must provide a
    validated map; absent frames fail explicitly rather than being resampled.
    """
    requested = tuple(requested_frames)
    if not requested or any(type(frame) is not int or frame < 1 for frame in requested):
        raise ValueError("Requested frame numbers must be positive integers")
    sidecar = Path(directory) / FRAME_MAP_NAME
    if not sidecar.exists():
        if max(requested) > frame_count:
            raise RuntimeError(
                f"{directory}: {frame_count} stored frames cannot supply original "
                f"frame {max(requested)}; selected trajectories need {FRAME_MAP_NAME}"
            )
        return tuple(frame - 1 for frame in requested)
    metadata = json.loads(sidecar.read_text())
    stored = metadata.get("source_frames_one_based")
    original_count = metadata.get("source_frame_count")
    if (
        metadata.get("schema") != FRAME_MAP_SCHEMA
        or type(original_count) is not int
        or not isinstance(stored, list)
        or len(stored) != frame_count
        or any(type(frame) is not int or frame < 1 or frame > original_count for frame in stored)
        or stored != sorted(set(stored))
    ):
        raise RuntimeError(f"{sidecar}: invalid selected-frame mapping")
    lookup = {frame: index for index, frame in enumerate(stored)}
    missing = [frame for frame in requested if frame not in lookup]
    if missing:
        raise RuntimeError(f"{directory}: original frames are not in this release: {missing[:5]}")
    return tuple(lookup[frame] for frame in requested)
