"""Helper tool: click on a video frame to collect pixel coordinates for georef.json.

Usage:
    uv run python tools/pick_points.py --video football-drone.mp4
    uv run python tools/pick_points.py --video football-drone.mp4 --frame 50

Click points on the image. Each click prints the pixel coordinate.
Press 'q' to quit and print the collected points as JSON.
"""

import argparse
import json
import sys

import cv2

points: list[dict] = []


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = len(points) + 1
        points.append({"pixel": [x, y], "gps": [0.0, 0.0], "label": f"point_{idx}"})
        print(f"  Point {idx}: pixel [{x}, {y}]  — fill in GPS coords in the output")

        # Draw marker on the image
        frame = param["frame"]
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(frame, str(idx), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Pick Reference Points (click to add, q to finish)", frame)


def main():
    parser = argparse.ArgumentParser(description="Click on a video frame to collect pixel coordinates for georef.")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to display (default: 0)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: print to stdout)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open {args.video}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Cannot read frame {args.frame}", file=sys.stderr)
        sys.exit(1)

    print(f"Showing frame {args.frame} from {args.video}")
    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print("Click to mark reference points. Press 'q' when done.\n")

    cv2.imshow("Pick Reference Points (click to add, q to finish)", frame)
    cv2.setMouseCallback("Pick Reference Points (click to add, q to finish)", on_click, {"frame": frame})

    while True:
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    if not points:
        print("No points selected.")
        return

    result = {"reference_points": points}
    output_json = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"\nSaved {len(points)} points to {args.output}")
        print("Edit the file to fill in the GPS [lat, lon] values for each point.")
    else:
        print(f"\n--- georef.json ({len(points)} points) ---")
        print(output_json)
        print("\nCopy the above into a georef.json file and fill in the GPS [lat, lon] values.")


if __name__ == "__main__":
    main()
