"""CLI entry point for video-segment: Gemma 4 + SAM3 local video analysis."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Local video segmentation: Gemma 4 analyzes scenes, SAM3 segments and tracks objects."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=str, help="Path to input video file (mp4)")
    source.add_argument("--camera", action="store_true", help="Use live camera feed")
    source.add_argument("--stream", type=str, metavar="URL",
                        help="RTSP/HTTP stream URL (e.g. rtsp://192.168.1.10:554/stream). "
                             "Passed directly to OpenCV VideoCapture.")

    parser.add_argument("--query", type=str, default=None,
                        help="What to find. Comma-separated for multiple labels: "
                             "'person, backpack, server rack'. Omit for Gemma auto-detection.")
    parser.add_argument("--output", type=str, default=None, help="Output video/JSON path. Auto-generated if omitted.")
    parser.add_argument("--threshold", type=float, default=0.3, help="SAM3 detection confidence threshold (default: 0.3)")
    parser.add_argument("--every-n-frames", type=int, default=2, help="Base detection interval in frames (default: 2). Adapted per-frame when --adaptive is on.")
    parser.add_argument("--inference-width", type=int, default=640, help="Downscale frames to this width for inference (default: 640). Use 0 for full resolution.")
    parser.add_argument("--scene-threshold", type=float, default=0.92, help="Scene-change sensitivity (0-1). Lower = more re-detections. (default: 0.92)")
    parser.add_argument("--max-reuse", type=int, default=60, help="Max frames to reuse a detection before forcing re-detect (default: 60)")
    parser.add_argument("--no-display", action="store_true", help="Disable live preview window")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive detection frequency (use fixed --every-n-frames instead)")

    # Entity modeling mode
    parser.add_argument("--entities", action="store_true", help="Build a JSON entity model with states instead of annotated video.")
    parser.add_argument("--state-every-n", type=int, default=30, help="Analyze entity states with Gemma every N frames (default: 30). Only used with --entities.")
    parser.add_argument("--max-entities", type=int, default=10, help="Max entities to analyze per state frame (default: 10). Only used with --entities.")
    parser.add_argument("--georef", type=str, default=None, help="Path to georef JSON with pixel→GPS reference points. Only used with --entities.")
    parser.add_argument("--stabilize", action="store_true", help="Compensate camera drift in motion trails (recommended for drone footage)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing pass-1 checkpoint and re-run from scratch.")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export (only write JSON).")
    parser.add_argument("--no-coco", action="store_true", help="Skip COCO JSON export.")

    args = parser.parse_args()

    inf_width = args.inference_width if args.inference_width > 0 else 999999
    adaptive = not args.no_adaptive

    # Parse comma-separated labels from --query
    if args.query:
        labels = [q.strip() for q in args.query.split(",") if q.strip()]
        query_arg = args.query  # kept for single-label paths that accept a string
    else:
        labels = None
        query_arg = None

    if args.entities:
        if not args.video:
            parser.error("--entities requires --video")

        from pipeline.tracker import process_video_entities

        json_path, video_path = process_video_entities(
            video_path=args.video,
            output_json=args.output,
            labels=labels,
            threshold=args.threshold,
            every_n=args.every_n_frames,
            state_every_n=args.state_every_n,
            inference_width=inf_width,
            max_entities_per_frame=args.max_entities,
            georef_path=args.georef,
            display=not args.no_display,
            stabilize=args.stabilize,
            adaptive=adaptive,
            scene_threshold=args.scene_threshold,
            max_reuse=args.max_reuse,
            resume=not args.no_resume,
            export_csv=not args.no_csv,
            export_coco=not args.no_coco,
        )
        print(f"Done. Entity model: {json_path}")
        print(f"Done. Tracked video: {video_path}")

    elif args.video:
        from pipeline.tracker import process_video

        output = process_video(
            video_path=args.video,
            output_path=args.output,
            labels=labels,
            threshold=args.threshold,
            every_n=args.every_n_frames,
            display=not args.no_display,
            inference_width=inf_width,
            scene_threshold=args.scene_threshold,
            max_reuse=args.max_reuse,
            stabilize=args.stabilize,
            adaptive=adaptive,
        )
        print(f"Done. Output: {output}")

    elif args.camera or args.stream:
        from pipeline.tracker import process_camera

        # --stream accepts a URL string; --camera uses an integer device ID
        camera_source = args.stream if args.stream else args.camera_id

        process_camera(
            labels=labels,
            threshold=args.threshold,
            every_n=args.every_n_frames,
            camera_source=camera_source,
            output_path=args.output,
            inference_width=inf_width,
            scene_threshold=args.scene_threshold,
            max_reuse=args.max_reuse,
            adaptive=adaptive,
            stabilize=args.stabilize,
        )
        print("Stream session ended.")


if __name__ == "__main__":
    main()
