# save as: simple_run.py

"""Simplified runner for quick video analysis"""

import sys
from pathlib import Path
from video_extraction.video_frame_extractor import VideoFrameExtractor, run_detection_on_frames, AnnotationUI, export_annotated_data

def simple_run(video_path: str, fps: float = 1.0, model: str = "yolov5"):
    """
    Simple one-function runner
    
    Args:
        video_path: Path to video file
        fps: Frames per second to extract
        model: Detection model to use
    """
    
    print(f"Processing video: {video_path}")
    print(f"Frames per second: {fps}")
    print(f"Detection model: {model}")
    
    # Step 1: Extract frames
    print("\n[1/4] Extracting frames...")
    extractor = VideoFrameExtractor(video_path)
    frames_dir, metadata = extractor.extract_frames(frames_per_second=fps)
    
    # Step 2: Run detection
    print("\n[2/4] Running object detection...")
    detections = run_detection_on_frames(frames_dir, model_type=model)
    
    # Step 3: Annotate
    print("\n[3/4] Manual annotation...")
    response = input("Launch annotation UI? (y/n): ")
    if response.lower() == 'y':
        ui = AnnotationUI(frames_dir)
        ui.run()
    
    # Step 4: Export
    print("\n[4/4] Exporting data...")
    export_annotated_data(frames_dir)
    
    print(f"\nComplete! Results in: {frames_dir}")
    return frames_dir

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_run.py <video_path> [fps] [model]")
        print("Example: python simple_run.py video.mp4 2 yolov5")
        sys.exit(1)
    
    video_path = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    model = sys.argv[3] if len(sys.argv) > 3 else "yolov5"
    
    simple_run(video_path, fps, model)