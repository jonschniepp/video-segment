"""Data model for tracked entities and their states over time."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EntitySnapshot:
    """A single observation of an entity at a specific frame."""
    frame: int
    timestamp: float  # seconds
    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float
    states: list[str]  # e.g., ["running", "near goalpost", "wearing red"]
    gps: list[float] | None = None  # [lat, lon] if georeferenced


@dataclass
class Entity:
    """A tracked entity with a persistent ID and timeline of observations."""
    entity_id: str  # e.g., "person_001"
    label: str  # e.g., "person"
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    timeline: list[EntitySnapshot] = field(default_factory=list)

    def add_snapshot(self, snapshot: EntitySnapshot):
        self.timeline.append(snapshot)
        if not self.first_seen_frame or snapshot.frame < self.first_seen_frame:
            self.first_seen_frame = snapshot.frame
        if snapshot.frame > self.last_seen_frame:
            self.last_seen_frame = snapshot.frame

    @property
    def current_states(self) -> list[str]:
        """Most recent states."""
        if not self.timeline:
            return []
        return self.timeline[-1].states


@dataclass
class SceneModel:
    """Complete scene model: all entities and their state timelines."""
    video_source: str
    fps: float
    total_frames: int
    frames_analyzed: int = 0
    entities: dict[str, Entity] = field(default_factory=dict)

    def add_or_update_entity(self, entity: Entity):
        self.entities[entity.entity_id] = entity

    def to_dict(self) -> dict:
        return {
            "video_source": self.video_source,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "frames_analyzed": self.frames_analyzed,
            "entity_count": len(self.entities),
            "entities": {
                eid: {
                    "entity_id": e.entity_id,
                    "label": e.label,
                    "first_seen_frame": e.first_seen_frame,
                    "last_seen_frame": e.last_seen_frame,
                    "snapshot_count": len(e.timeline),
                    "current_states": e.current_states,
                    "timeline": [
                        {
                            "frame": s.frame,
                            "timestamp": round(s.timestamp, 3),
                            "bbox": [round(v, 1) for v in s.bbox],
                            "confidence": round(s.confidence, 3),
                            "states": s.states,
                            **({"gps": [round(v, 7) for v in s.gps]} if s.gps else {}),
                        }
                        for s in e.timeline
                    ],
                }
                for eid, e in self.entities.items()
            },
        }

    def save(self, path: str):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        print(f"Scene model saved to: {path}")
