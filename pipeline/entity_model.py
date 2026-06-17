"""Data model for tracked entities and their states over time."""

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class StateTransition:
    """A detected change in entity state."""
    frame: int
    timestamp: float
    from_states: list[str]
    to_states: list[str]


@dataclass
class EntitySnapshot:
    """A single observation of an entity at a specific frame."""
    frame: int
    timestamp: float  # seconds
    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float
    states: list[str]  # e.g., ["running", "near goalpost", "wearing red"]
    gps: list[float] | None = None  # [lat, lon] if georeferenced
    vx: float | None = None  # pixels/sec horizontal velocity
    vy: float | None = None  # pixels/sec vertical velocity


@dataclass
class Entity:
    """A tracked entity with a persistent ID and timeline of observations."""
    entity_id: str  # e.g., "person_001"
    label: str  # e.g., "person"
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    timeline: list[EntitySnapshot] = field(default_factory=list)
    transitions: list[StateTransition] = field(default_factory=list)

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

    def detect_transitions(self):
        """Post-process timeline to find state changes.

        Compares consecutive snapshots that have non-empty states and records
        transitions when the state set changes meaningfully.
        """
        self.transitions = []
        prev_snap = None
        for snap in self.timeline:
            if not snap.states or snap.states == ["not analyzed"]:
                continue
            if prev_snap is not None:
                prev_set = set(prev_snap.states)
                curr_set = set(snap.states)
                if prev_set != curr_set:
                    self.transitions.append(StateTransition(
                        frame=snap.frame,
                        timestamp=snap.timestamp,
                        from_states=list(prev_set),
                        to_states=list(curr_set),
                    ))
            prev_snap = snap


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

    def compute_all_transitions(self):
        """Run state transition detection across all entities."""
        for entity in self.entities.values():
            entity.detect_transitions()

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
                    "transitions": [
                        {
                            "frame": t.frame,
                            "timestamp": round(t.timestamp, 3),
                            "from": t.from_states,
                            "to": t.to_states,
                        }
                        for t in e.transitions
                    ],
                    "timeline": [
                        {
                            "frame": s.frame,
                            "timestamp": round(s.timestamp, 3),
                            "bbox": [round(v, 1) for v in s.bbox],
                            "confidence": round(s.confidence, 3),
                            "states": s.states,
                            **({"gps": [round(v, 7) for v in s.gps]} if s.gps else {}),
                            **({"vx": round(s.vx, 2), "vy": round(s.vy, 2)}
                               if s.vx is not None else {}),
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

    def save_csv(self, path: str):
        """Export entity timeline as a flat CSV file.

        One row per snapshot. Columns: entity_id, label, frame, timestamp,
        x1, y1, x2, y2, confidence, states, vx, vy, lat, lon.
        """
        rows = []
        for eid, entity in self.entities.items():
            for snap in entity.timeline:
                x1, y1, x2, y2 = snap.bbox
                rows.append({
                    "entity_id": eid,
                    "label": entity.label,
                    "frame": snap.frame,
                    "timestamp": round(snap.timestamp, 3),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                    "confidence": round(snap.confidence, 3),
                    "states": "|".join(snap.states),
                    "vx": round(snap.vx, 2) if snap.vx is not None else "",
                    "vy": round(snap.vy, 2) if snap.vy is not None else "",
                    "lat": round(snap.gps[0], 7) if snap.gps else "",
                    "lon": round(snap.gps[1], 7) if snap.gps else "",
                })

        if not rows:
            return

        fieldnames = list(rows[0].keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        Path(path).write_text(buf.getvalue())
        print(f"CSV exported to: {path}")

    def save_coco(self, path: str):
        """Export detections in COCO JSON format.

        Produces a detection-style COCO file: images list (one per keyframe),
        annotations list (one per entity snapshot), and categories derived
        from entity labels.
        """
        # Collect unique labels -> category IDs
        labels = sorted({e.label for e in self.entities.values()})
        label_to_cat = {lbl: i + 1 for i, lbl in enumerate(labels)}
        categories = [{"id": cid, "name": lbl, "supercategory": "object"}
                      for lbl, cid in label_to_cat.items()]

        # Collect all keyframes referenced in any snapshot
        frame_set = set()
        for entity in self.entities.values():
            for snap in entity.timeline:
                frame_set.add(snap.frame)

        frame_ids = {f: i + 1 for i, f in enumerate(sorted(frame_set))}
        images = [
            {
                "id": img_id,
                "frame": frame,
                "file_name": f"frame_{frame:06d}.jpg",
                "fps": self.fps,
            }
            for frame, img_id in sorted(frame_ids.items())
        ]

        annotations = []
        ann_id = 1
        for eid, entity in self.entities.items():
            cat_id = label_to_cat[entity.label]
            for snap in entity.timeline:
                x1, y1, x2, y2 = snap.bbox
                w = x2 - x1
                h = y2 - y1
                annotations.append({
                    "id": ann_id,
                    "image_id": frame_ids[snap.frame],
                    "category_id": cat_id,
                    "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                    "area": round(w * h, 1),
                    "score": round(snap.confidence, 3),
                    "entity_id": eid,
                    "iscrowd": 0,
                })
                ann_id += 1

        coco = {
            "info": {"video_source": self.video_source, "fps": self.fps},
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
        Path(path).write_text(json.dumps(coco, indent=2))
        print(f"COCO JSON exported to: {path}")
