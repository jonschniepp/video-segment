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
    parser.add_argument("--output", type=str, default=None, help="Output video path. Auto-generated if omitted.")
    parser.add_argument("--threshold", type=float, default=0.3, help="SAM3 detection confidence threshold (default: 0.3)")
    parser.add_argument("--every-n-frames", type=int, default=2, help="Run detection every N frames (default: 2)")
    parser.add_argument("--no-display", action="store_true", help="Disable live preview window")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID (default: 0)")

    args = parser.parse_args()

    from pipeline.tracker import process_camera, process_video

    if args.video:
        output = process_video(
            video_path=args.video,
            output_path=args.output,
            query=args.query,
            threshold=args.threshold,
            every_n=args.every_n_frames,
            display=not args.no_display,
        )
        print(f"Done. Output: {output}")

    elif args.camera:
        process_camera(
            query=args.query,
            threshold=args.threshold,
            every_n=args.every_n_frames,
            camera_id=args.camera_id,
            output_path=args.output,
        )
        print("Camera session ended.")


if __name__ == "__main__":
    main()
