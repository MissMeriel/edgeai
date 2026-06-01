# save as: run_video_analysis.py

import argparse
import sys
from pathlib import Path
import json
import cv2
from video_extraction.video_frame_extractor import (
    VideoFrameExtractor,
    ObjectDetector,
    run_detection_on_frames,
    AnnotationUI,
    export_annotated_data
)

def main():
    """Main runner for video analysis pipeline"""
    
    parser = argparse.ArgumentParser(
        description="Extract frames from video, run object detection, and annotate objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - extract 1 frame per second
  python run_video_analysis.py input_video.mp4

  # Extract 2 frames per second
  python run_video_analysis.py input_video.mp4 --fps 2

  # Extract frames only (no detection)
  python run_video_analysis.py input_video.mp4 --extract-only

  # Run detection only on existing frames
  python run_video_analysis.py --detect-only --frames-dir extracted_frames/video_20240101_120000_abc123

  # Annotate existing frames
  python run_video_analysis.py --annotate-only --frames-dir extracted_frames/video_20240101_120000_abc123

  # Full pipeline with YOLOv8
  python run_video_analysis.py input_video.mp4 --fps 2 --model yolov8 --confidence 0.6

  # Extract specific time range
  python run_video_analysis.py input_video.mp4 --start-time 10 --end-time 60 --fps 1
        """
    )
    
    # Input/Output arguments
    parser.add_argument('video_path', nargs='?', help='Path to input video file')
    parser.add_argument('--output-dir', type=str, default='extracted_frames',
                       help='Base output directory (default: extracted_frames)')
    parser.add_argument('--frames-dir', type=str,
                       help='Existing frames directory (for detect-only or annotate-only modes)')
    
    # Extraction arguments
    parser.add_argument('--fps', type=float, default=1.0,
                       help='Frames to extract per second (default: 1.0)')
    parser.add_argument('--start-time', type=float, default=0,
                       help='Start time in seconds (default: 0)')
    parser.add_argument('--end-time', type=float,
                       help='End time in seconds (default: end of video)')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum number of frames to extract')
    
    # Detection arguments
    parser.add_argument('--model', type=str, default='yolov5',
                       choices=['yolov5', 'yolov8', 'faster-rcnn', 'clip'],
                       help='Object detection model (default: yolov5)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run detection on (default: cpu)')
    
    # Pipeline control
    parser.add_argument('--extract-only', action='store_true',
                       help='Only extract frames, skip detection and annotation')
    parser.add_argument('--detect-only', action='store_true',
                       help='Only run detection on existing frames')
    parser.add_argument('--annotate-only', action='store_true',
                       help='Only run annotation UI on existing frames')
    parser.add_argument('--skip-detection', action='store_true',
                       help='Skip automatic detection step')
    parser.add_argument('--skip-annotation', action='store_true',
                       help='Skip manual annotation UI')
    parser.add_argument('--auto-export', action='store_true',
                       help='Automatically export data without annotation')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.detect_only or args.annotate_only:
        if not args.frames_dir:
            parser.error("--frames-dir is required for --detect-only or --annotate-only")
        if not Path(args.frames_dir).exists():
            parser.error(f"Frames directory does not exist: {args.frames_dir}")
    else:
        if not args.video_path:
            parser.error("video_path is required unless using --detect-only or --annotate-only")
        if not Path(args.video_path).exists():
            parser.error(f"Video file not found: {args.video_path}")
    
    print("="*70)
    print("VIDEO ANALYSIS PIPELINE")
    print("="*70)
    
    # ========================================================================
    # STEP 1: EXTRACT FRAMES
    # ========================================================================
    
    if not (args.detect_only or args.annotate_only):
        print(f"\n{'='*70}")
        print("STEP 1: EXTRACTING FRAMES FROM VIDEO")
        print(f"{'='*70}\n")
        
        try:
            extractor = VideoFrameExtractor(
                video_path=args.video_path,
                output_base_dir=args.output_dir
            )
            
            frames_dir, metadata = extractor.extract_frames(
                frames_per_second=args.fps,
                start_time=args.start_time,
                end_time=args.end_time,
                max_frames=args.max_frames
            )
            
            print(f"\n✓ Frame extraction complete!")
            print(f"  Output directory: {frames_dir}")
            
            if args.extract_only:
                print("\n--extract-only flag set. Exiting.")
                return
            
        except Exception as e:
            print(f"\n✗ Error during frame extraction: {e}")
            sys.exit(1)
    else:
        frames_dir = Path(args.frames_dir)
        print(f"\nUsing existing frames directory: {frames_dir}")
    
    # ========================================================================
    # STEP 2: RUN OBJECT DETECTION
    # ========================================================================
    
    if not (args.annotate_only or args.skip_detection):
        print(f"\n{'='*70}")
        print("STEP 2: RUNNING OBJECT DETECTION")
        print(f"{'='*70}\n")
        
        try:
            detections = run_detection_on_frames(
                frames_dir=frames_dir,
                model_type=args.model,
                confidence_threshold=args.confidence,
                device=args.device
            )
            
            print(f"\n✓ Object detection complete!")
            
        except Exception as e:
            print(f"\n✗ Error during detection: {e}")
            print("Continuing to annotation step...")
    
    # ========================================================================
    # STEP 3: MANUAL ANNOTATION UI
    # ========================================================================
    
    if not (args.skip_annotation or args.auto_export):
        print(f"\n{'='*70}")
        print("STEP 3: MANUAL ANNOTATION")
        print(f"{'='*70}\n")
        
        response = input("Launch annotation UI? (y/n): ")
        if response.lower() == 'y':
            try:
                ui = AnnotationUI(frames_dir=frames_dir)
                ui.run()
                print(f"\n✓ Annotation complete!")
            except Exception as e:
                print(f"\n✗ Error during annotation: {e}")
        else:
            print("Skipping annotation UI.")
    
    # ========================================================================
    # STEP 4: EXPORT ANNOTATED DATA
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("STEP 4: EXPORTING ANNOTATED DATA")
    print(f"{'='*70}\n")
    
    try:
        export_annotated_data(frames_dir=frames_dir)
        print(f"\n✓ Data export complete!")
    except Exception as e:
        print(f"\n✗ Error during export: {e}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    
    print(f"Output directory: {frames_dir}")
    print(f"\nGenerated files:")
    print(f"  - metadata.json              : Frame extraction metadata")
    print(f"  - detections.json            : Auto-detected objects")
    print(f"  - manual_annotations.json    : Manually annotated objects")
    print(f"  - classes.txt                : List of object classes")
    print(f"  - annotated_export/          : Exported data in various formats")
    print(f"    - combined_annotations.json: All annotations combined")
    print(f"    - yolo_format/              : YOLO training format")
    print(f"    - coco_format/              : COCO format")
    print(f"    - visualization/            : Annotated images")
    
    print(f"\nNext steps:")
    print(f"  1. Review annotations in FiftyOne:")
    print(f"     python view_in_fiftyone.py {frames_dir}")
    print(f"  2. Train a model with the annotations")
    print(f"  3. Run inference on new videos")

if __name__ == "__main__":
    main()