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

    parser.add_argument("--query", type=str, default=None, help="What to find (e.g., 'person', 'find all cars'). Omit for auto-detection.")
    parser.add_argument("--output", type=str, default=None, help="Output video/JSON path. Auto-generated if omitted.")
    parser.add_argument("--threshold", type=float, default=0.3, help="SAM3 detection confidence threshold (default: 0.3)")
    parser.add_argument("--every-n-frames", type=int, default=2, help="Run detection every N frames (default: 2)")
    parser.add_argument("--inference-width", type=int, default=640, help="Downscale frames to this width for inference (default: 640). Use 0 for full resolution.")
    parser.add_argument("--scene-threshold", type=float, default=0.92, help="Scene-change sensitivity (0-1). Lower = more re-detections. (default: 0.92)")
    parser.add_argument("--max-reuse", type=int, default=60, help="Max frames to reuse a detection before forcing re-detect (default: 60)")
    parser.add_argument("--no-display", action="store_true", help="Disable live preview window")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID (default: 0)")

    # Entity modeling mode
    parser.add_argument("--entities", action="store_true", help="Build a JSON entity model with states instead of annotated video.")
    parser.add_argument("--state-every-n", type=int, default=30, help="Analyze entity states with Gemma every N frames (default: 30). Only used with --entities.")
    parser.add_argument("--max-entities", type=int, default=10, help="Max entities to analyze per state frame (default: 10). Only used with --entities.")
    parser.add_argument("--georef", type=str, default=None, help="Path to georef JSON with pixel→GPS reference points. Only used with --entities.")

    args = parser.parse_args()

    inf_width = args.inference_width if args.inference_width > 0 else 999999

    if args.entities:
        if not args.video:
            parser.error("--entities requires --video")

        from pipeline.tracker import process_video_entities

        json_path, video_path = process_video_entities(
            video_path=args.video,
            output_json=args.output,
            query=args.query,
            threshold=args.threshold,
            every_n=args.every_n_frames,
            state_every_n=args.state_every_n,
            inference_width=inf_width,
            max_entities_per_frame=args.max_entities,
            georef_path=args.georef,
            display=not args.no_display,
        )
        print(f"Done. Entity model: {json_path}")
        print(f"Done. Tracked video: {video_path}")

    elif args.video:
        from pipeline.tracker import process_video

        output = process_video(
            video_path=args.video,
            output_path=args.output,
            query=args.query,
            threshold=args.threshold,
            every_n=args.every_n_frames,
            display=not args.no_display,
            inference_width=inf_width,
            scene_threshold=args.scene_threshold,
            max_reuse=args.max_reuse,
        )
        print(f"Done. Output: {output}")

    elif args.camera:
        from pipeline.tracker import process_camera

        process_camera(
            query=args.query,
            threshold=args.threshold,
            every_n=args.every_n_frames,
            camera_id=args.camera_id,
            output_path=args.output,
            inference_width=inf_width,
        )
        print("Camera session ended.")


if __name__ == "__main__":
    main()
