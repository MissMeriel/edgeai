# save as: video_frame_extractor.py (UPDATED VERSION)

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='QFont::setPointSizeF.*')
import os
import sys

# Define a context manager to redirect stderr
class DevNull:
    def write(self, *args, **kwargs):
        pass

# Use the context manager to suppress the message during specific operations
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Redirect stderr
    old_stderr = sys.stderr
    sys.stderr = DevNull()
    try:
        import cv2
        # ... your OpenCV code that produces the warning ...
    finally:
        # Restore stderr
        sys.stderr = old_stderr
        
import cv2
import os
from pathlib import Path
import hashlib
import time
from datetime import datetime
from typing import Optional, Tuple, List
import json
from tqdm import tqdm
import numpy as np
import subprocess
import tempfile

class VideoFrameExtractor:
    """Extract frames from video at specified rate with MPG support"""
    
    def __init__(self, video_path: str, output_base_dir: str = "extracted_frames"):
        """
        Initialize video frame extractor
        
        Args:
            video_path: Path to video file
            output_base_dir: Base directory for output
        """
        self.video_path = video_path
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate video
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check if ffmpeg is available (better for problematic formats)
        self.has_ffmpeg = self._check_ffmpeg()
        
        # Try to open video with OpenCV first
        self.use_opencv = True
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError("Cannot open with OpenCV")
            
            # Test read a frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise ValueError("Cannot read frames with OpenCV")
            
            # Reset to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        except Exception as e:
            print(f"⚠ OpenCV cannot handle this video: {e}")
            if self.has_ffmpeg:
                print("→ Will use FFmpeg for extraction")
                self.use_opencv = False
                self.cap = None
            else:
                raise ValueError(
                    f"Cannot open video with OpenCV and FFmpeg is not available. "
                    f"Install FFmpeg: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)"
                )
        
        # Get video properties
        if self.use_opencv:
            self._get_video_properties_opencv()
        else:
            self._get_video_properties_ffmpeg()
        
        print(f"Video Info:")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Total Frames: {self.total_frames}")
        print(f"  Duration: {self.duration:.2f}s")
        print(f"  Extraction method: {'OpenCV' if self.use_opencv else 'FFmpeg'}")
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _get_video_properties_opencv(self):
        """Get video properties using OpenCV"""
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
    
    def _get_video_properties_ffmpeg(self):
        """Get video properties using FFmpeg"""
        try:
            # Use ffprobe to get video information
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
                '-of', 'json',
                self.video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise ValueError(f"FFprobe error: {result.stderr}")
            
            data = json.loads(result.stdout)
            stream = data['streams'][0]
            
            self.width = int(stream['width'])
            self.height = int(stream['height'])
            
            # Parse frame rate (can be "30000/1001" format)
            fps_str = stream['r_frame_rate']
            num, den = map(int, fps_str.split('/'))
            self.fps = num / den
            
            # Get total frames and duration
            if 'nb_frames' in stream:
                self.total_frames = int(stream['nb_frames'])
            elif 'duration' in stream:
                self.duration = float(stream['duration'])
                self.total_frames = int(self.duration * self.fps)
            else:
                # Fallback: count frames (slower)
                self.total_frames = self._count_frames_ffmpeg()
            
            if 'duration' in stream:
                self.duration = float(stream['duration'])
            else:
                self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            
        except Exception as e:
            print(f"Error getting video properties: {e}")
            # Set defaults
            self.width = 640
            self.height = 480
            self.fps = 30.0
            self.total_frames = 1000
            self.duration = self.total_frames / self.fps
    
    def _count_frames_ffmpeg(self) -> int:
        """Count total frames using ffmpeg (slow but accurate)"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-count_frames',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames',
                '-of', 'default=nokey=1:noprint_wrappers=1',
                self.video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return int(result.stdout.strip())
        except:
            pass
        
        return 1000  # Default fallback
    
    def extract_frames(self, 
                      frames_per_second: float = 1.0,
                      output_dir: Optional[str] = None,
                      start_time: float = 0,
                      end_time: Optional[float] = None,
                      max_frames: Optional[int] = None) -> Tuple[Path, dict]:
        """
        Extract frames from video
        
        Args:
            frames_per_second: Number of frames to extract per second
            output_dir: Custom output directory (default: auto-generated)
            start_time: Start time in seconds
            end_time: End time in seconds (None = end of video)
            max_frames: Maximum number of frames to extract
        
        Returns:
            Tuple of (output_directory, metadata_dict)
        """
        
        # Create output directory with timestamp and hash
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = Path(self.video_path).stem
            random_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            output_dir = self.output_base_dir / f"{video_name}_{timestamp}_{random_hash}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use appropriate extraction method
        if self.use_opencv:
            return self._extract_frames_opencv(
                frames_per_second, output_dir, start_time, end_time, max_frames
            )
        else:
            return self._extract_frames_ffmpeg(
                frames_per_second, output_dir, start_time, end_time, max_frames
            )
    
    def _extract_frames_opencv(self,
                              frames_per_second: float,
                              output_dir: Path,
                              start_time: float,
                              end_time: Optional[float],
                              max_frames: Optional[int]) -> Tuple[Path, dict]:
        """Extract frames using OpenCV with deinterlacing"""
        
        # Calculate frame interval
        frame_interval = int(self.fps / frames_per_second)
        if frame_interval < 1:
            frame_interval = 1
            frames_per_second = self.fps
            print(f"Warning: Adjusted to {frames_per_second:.2f} fps (max for this video)")
        
        # Calculate start and end frames
        start_frame = int(start_time * self.fps)
        if end_time is None:
            end_frame = self.total_frames
        else:
            end_frame = int(end_time * self.fps)
        
        # Set video position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames
        frame_count = 0
        saved_count = 0
        metadata = self._create_metadata_dict(
            output_dir, frames_per_second, frame_interval, start_time, end_time
        )
        
        total_to_extract = min(
            (end_frame - start_frame) // frame_interval,
            max_frames if max_frames else float('inf')
        )
        
        print(f"\nExtracting frames to: {output_dir}")
        print(f"Frame rate: {frames_per_second:.2f} fps")
        print(f"Estimated frames to extract: {int(total_to_extract)}")
        
        pbar = tqdm(total=int(total_to_extract), desc="Extracting frames")
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret or (end_frame and frame_count >= end_frame - start_frame):
                break
            
            if max_frames and saved_count >= max_frames:
                break
            
            # Save frame at interval
            if frame_count % frame_interval == 0:
                try:
                    # Apply deinterlacing if needed
                    frame = self._deinterlace_frame(frame)
                    
                    # Generate filename
                    current_frame_num = start_frame + frame_count
                    timestamp_sec = current_frame_num / self.fps
                    filename = f"frame_{saved_count:06d}_t{timestamp_sec:.2f}s.jpg"
                    filepath = output_dir / filename
                    
                    # Save frame
                    success = cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    if success:
                        # Store metadata
                        frame_metadata = {
                            'filename': filename,
                            'filepath': str(filepath),
                            'frame_number': current_frame_num,
                            'timestamp': timestamp_sec,
                            'extracted_index': saved_count
                        }
                        metadata['frames'].append(frame_metadata)
                        
                        saved_count += 1
                        pbar.update(1)
                except Exception as e:
                    print(f"\nWarning: Failed to save frame {saved_count}: {e}")
            
            frame_count += 1
        
        pbar.close()
        
        return self._finalize_extraction(output_dir, metadata, saved_count)
    
    def _extract_frames_ffmpeg(self,
                              frames_per_second: float,
                              output_dir: Path,
                              start_time: float,
                              end_time: Optional[float],
                              max_frames: Optional[int]) -> Tuple[Path, dict]:
        """Extract frames using FFmpeg (better for problematic formats)"""
        
        metadata = self._create_metadata_dict(
            output_dir, frames_per_second, 0, start_time, end_time
        )
        
        print(f"\nExtracting frames to: {output_dir}")
        print(f"Frame rate: {frames_per_second:.2f} fps")
        print("Using FFmpeg for extraction (handles interlaced video)...")
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-i', self.video_path,
            '-ss', str(start_time),  # Start time
        ]
        
        # Add end time if specified
        if end_time is not None:
            cmd.extend(['-t', str(end_time - start_time)])
        
        # Video filters: deinterlace and set frame rate
        vf_filters = [
            'yadif=mode=send_frame:parity=auto:deint=all',  # Deinterlace
            f'fps={frames_per_second}'  # Set output fps
        ]
        
        cmd.extend([
            '-vf', ','.join(vf_filters),
            '-q:v', '2',  # High quality
            '-frame_pts', '1',  # Use presentation timestamp
        ])
        
        # Limit number of frames if specified
        if max_frames:
            cmd.extend(['-frames:v', str(max_frames)])
        
        # Output pattern
        output_pattern = str(output_dir / 'frame_%06d.jpg')
        cmd.append(output_pattern)
        
        try:
            # Run FFmpeg
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"FFmpeg stderr: {result.stderr}")
                raise ValueError(f"FFmpeg extraction failed: {result.stderr}")
            
            # Count extracted frames and build metadata
            extracted_files = sorted(output_dir.glob("frame_*.jpg"))
            saved_count = len(extracted_files)
            
            print(f"\nPost-processing {saved_count} frames...")
            for idx, filepath in enumerate(tqdm(extracted_files, desc="Building metadata")):
                timestamp_sec = idx / frames_per_second + start_time
                
                # Rename with timestamp
                new_filename = f"frame_{idx:06d}_t{timestamp_sec:.2f}s.jpg"
                new_filepath = output_dir / new_filename
                filepath.rename(new_filepath)
                
                frame_metadata = {
                    'filename': new_filename,
                    'filepath': str(new_filepath),
                    'frame_number': int(timestamp_sec * self.fps),
                    'timestamp': timestamp_sec,
                    'extracted_index': idx
                }
                metadata['frames'].append(frame_metadata)
            
            return self._finalize_extraction(output_dir, metadata, saved_count)
            
        except subprocess.TimeoutExpired:
            raise ValueError("FFmpeg extraction timed out")
        except Exception as e:
            raise ValueError(f"FFmpeg extraction error: {e}")
    
    def _deinterlace_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply simple deinterlacing to frame"""
        try:
            # Simple bob deinterlacing: blend even and odd lines
            height, width = frame.shape[:2]
            
            # Check if frame appears interlaced (basic check)
            if height > 240:  # Only for reasonable resolutions
                # Create deinterlaced frame by averaging adjacent lines
                even_lines = frame[::2, :]
                odd_lines = frame[1::2, :]
                
                # Resize to same dimensions
                even_resized = cv2.resize(even_lines, (width, height), interpolation=cv2.INTER_LINEAR)
                odd_resized = cv2.resize(odd_lines, (width, height), interpolation=cv2.INTER_LINEAR)
                
                # Blend
                frame = cv2.addWeighted(even_resized, 0.5, odd_resized, 0.5, 0)
            
            return frame
        except:
            return frame  # Return original on error
    
    def _create_metadata_dict(self,
                             output_dir: Path,
                             frames_per_second: float,
                             frame_interval: int,
                             start_time: float,
                             end_time: Optional[float]) -> dict:
        """Create metadata dictionary"""
        return {
            'video_path': self.video_path,
            'video_name': Path(self.video_path).name,
            'output_dir': str(output_dir),
            'extraction_time': datetime.now().isoformat(),
            'frames_per_second': frames_per_second,
            'video_fps': self.fps,
            'frame_interval': frame_interval,
            'start_time': start_time,
            'end_time': end_time if end_time else self.duration,
            'video_resolution': f"{self.width}x{self.height}",
            'extraction_method': 'opencv' if self.use_opencv else 'ffmpeg',
            'frames': []
        }
    
    def _finalize_extraction(self, output_dir: Path, metadata: dict, saved_count: int) -> Tuple[Path, dict]:
        """Finalize extraction and save metadata"""
        
        # Save metadata
        metadata['total_frames_extracted'] = saved_count
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nExtraction complete!")
        print(f"Frames saved: {saved_count}")
        print(f"Output directory: {output_dir}")
        print(f"Metadata saved: {metadata_path}")
        
        return output_dir, metadata
    
    def __del__(self):
        """Release video capture"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


# ============================================================================
# HELPER FUNCTION: VIDEO FORMAT CHECKER
# ============================================================================

def check_video_format(video_path: str) -> dict:
    """
    Check video format and codec information
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video information
    """
    
    info = {
        'path': video_path,
        'exists': os.path.exists(video_path),
        'opencv_compatible': False,
        'ffmpeg_available': False,
        'properties': {}
    }
    
    if not info['exists']:
        return info
    
    # Check FFmpeg availability
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        info['ffmpeg_available'] = True
    except:
        pass
    
    # Try OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                info['opencv_compatible'] = True
                info['properties']['opencv'] = {
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC))
                }
        cap.release()
    except Exception as e:
        info['opencv_error'] = str(e)
    
    # Try FFprobe
    if info['ffmpeg_available']:
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_format',
                '-show_streams',
                '-of', 'json',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Extract video stream info
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        info['properties']['ffmpeg'] = {
                            'codec': stream.get('codec_name'),
                            'codec_long_name': stream.get('codec_long_name'),
                            'width': stream.get('width'),
                            'height': stream.get('height'),
                            'fps': stream.get('r_frame_rate'),
                            'pixel_format': stream.get('pix_fmt'),
                            'field_order': stream.get('field_order', 'progressive'),
                            'is_interlaced': stream.get('field_order', 'progressive') != 'progressive'
                        }
                        break
                
                info['properties']['format'] = {
                    'format_name': data.get('format', {}).get('format_name'),
                    'format_long_name': data.get('format', {}).get('format_long_name'),
                    'duration': data.get('format', {}).get('duration'),
                    'size': data.get('format', {}).get('size'),
                    'bit_rate': data.get('format', {}).get('bit_rate')
                }
        except Exception as e:
            info['ffprobe_error'] = str(e)
    
    return info


# ============================================================================
# HELPER FUNCTION: PRINT VIDEO INFO
# ============================================================================

def print_video_info(video_path: str):
    """Print detailed video information"""
    
    print("="*70)
    print(f"VIDEO FORMAT CHECK: {Path(video_path).name}")
    print("="*70)
    
    info = check_video_format(video_path)
    
    print(f"\nFile exists: {info['exists']}")
    
    if not info['exists']:
        print("✗ File not found!")
        return
    
    print(f"OpenCV compatible: {info['opencv_compatible']}")
    print(f"FFmpeg available: {info['ffmpeg_available']}")
    
    if 'ffmpeg' in info['properties']:
        print("\nFFmpeg Video Properties:")
        props = info['properties']['ffmpeg']
        print(f"  Codec: {props.get('codec')} ({props.get('codec_long_name')})")
        print(f"  Resolution: {props.get('width')}x{props.get('height')}")
        print(f"  FPS: {props.get('fps')}")
        print(f"  Pixel Format: {props.get('pixel_format')}")
        print(f"  Field Order: {props.get('field_order')}")
        print(f"  Is Interlaced: {props.get('is_interlaced')}")
    
    if 'format' in info['properties']:
        print("\nFormat Properties:")
        fmt = info['properties']['format']
        print(f"  Format: {fmt.get('format_name')}")
        print(f"  Description: {fmt.get('format_long_name')}")
        if fmt.get('duration'):
            print(f"  Duration: {float(fmt.get('duration')):.2f}s")
        if fmt.get('size'):
            size_mb = int(fmt.get('size')) / (1024 * 1024)
            print(f"  File Size: {size_mb:.2f} MB")
    
    if 'opencv' in info['properties']:
        print("\nOpenCV Properties:")
        props = info['properties']['opencv']
        print(f"  Resolution: {props.get('width')}x{props.get('height')}")
        print(f"  FPS: {props.get('fps')}")
        print(f"  Frame Count: {props.get('frame_count')}")
    
    # Recommendation
    print("\nRecommendation:")
    if info['opencv_compatible']:
        print("  ✓ Can use OpenCV for extraction")
    elif info['ffmpeg_available']:
        print("  ⚠ Should use FFmpeg for extraction (video has compatibility issues)")
        if info['properties'].get('ffmpeg', {}).get('is_interlaced'):
            print("  ℹ Video is interlaced - will apply deinterlacing")
    else:
        print("  ✗ Install FFmpeg to extract this video: sudo apt-get install ffmpeg")



# ============================================================================
# 2. OBJECT DETECTION INTEGRATION
# ============================================================================

class ObjectDetector:
    """Wrapper for various object detection models"""
    
    def __init__(self, model_type: str = "yolov5", device: str = "cpu"):
        """
        Initialize object detector
        
        Args:
            model_type: "yolov5", "yolov8", "faster-rcnn", or "clip"
            device: "cpu" or "cuda"
        """
        self.model_type = model_type
        self.device = device
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load detection model"""
        
        if self.model_type == "yolov5":
            print("Loading YOLOv5 model...")
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=self.device)
            
        elif self.model_type == "yolov8":
            print("Loading YOLOv8 model...")
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            
        elif self.model_type == "faster-rcnn":
            print("Loading Faster R-CNN model...")
            import torch
            import torchvision
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
        elif self.model_type == "clip":
            print("Loading CLIP model...")
            import torch
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def detect_objects(self, image_path: str, confidence_threshold: float = 0.5) -> list:
        """
        Detect objects in image
        
        Args:
            image_path: Path to image
            confidence_threshold: Minimum confidence score
        
        Returns:
            List of detections with format:
            [{'bbox': [x1, y1, x2, y2], 'label': str, 'confidence': float}, ...]
        """
        
        if self.model_type in ["yolov5", "yolov8"]:
            return self._detect_yolo(image_path, confidence_threshold)
        elif self.model_type == "faster-rcnn":
            return self._detect_faster_rcnn(image_path, confidence_threshold)
        elif self.model_type == "clip":
            return self._detect_clip(image_path, confidence_threshold)
    
    def _detect_yolo(self, image_path: str, threshold: float) -> list:
        """Detect with YOLO"""
        results = self.model(image_path)
        detections = []
        
        # Parse results
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf >= threshold:
                detections.append({
                    'bbox': [float(x) for x in box],
                    'label': results.names[int(cls)],
                    'confidence': float(conf)
                })
        
        return detections
    
    def _detect_faster_rcnn(self, image_path: str, threshold: float) -> list:
        """Detect with Faster R-CNN"""
        import torch
        import torchvision.transforms as T
        from PIL import Image
        
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Detect
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Parse results
        detections = []
        pred = predictions[0]
        
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            if score >= threshold:
                detections.append({
                    'bbox': box.cpu().numpy().tolist(),
                    'label': self._get_coco_label(int(label)),
                    'confidence': float(score)
                })
        
        return detections
    
    def _detect_clip(self, image_path: str, threshold: float) -> list:
        """Detect with CLIP (zero-shot classification)"""
        # CLIP is more for classification, returning empty list for detection
        # You would need to implement sliding window or use CLIP for classification
        return []
    
    def _get_coco_label(self, label_id: int) -> str:
        """Get COCO label name from ID"""
        COCO_LABELS = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        if 0 <= label_id < len(COCO_LABELS):
            return COCO_LABELS[label_id]
        return "unknown"

# ============================================================================
# 3. BATCH DETECTION ON EXTRACTED FRAMES
# ============================================================================

def run_detection_on_frames(frames_dir: Path, 
                            model_type: str = "yolov5",
                            confidence_threshold: float = 0.5,
                            device: str = "cpu") -> dict:
    """
    Run object detection on all frames
    
    Args:
        frames_dir: Directory containing frames
        model_type: Type of detection model
        confidence_threshold: Minimum confidence
        device: cpu or cuda
    
    Returns:
        Dictionary mapping filenames to detections
    """
    
    print(f"\n{'='*70}")
    print("RUNNING OBJECT DETECTION")
    print(f"{'='*70}")
    print(f"Model: {model_type}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Device: {device}")
    
    # Initialize detector
    detector = ObjectDetector(model_type=model_type, device=device)
    
    # Get all image files
    image_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    
    # Run detection
    results = {}
    
    print(f"\nProcessing {len(image_files)} images...")
    for image_path in tqdm(image_files, desc="Detecting objects"):
        detections = detector.detect_objects(str(image_path), confidence_threshold)
        results[image_path.name] = detections
    
    # Save detection results
    results_path = frames_dir / "detections.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetection complete!")
    print(f"Results saved: {results_path}")
    
    # Print summary
    total_detections = sum(len(dets) for dets in results.values())
    images_with_detections = sum(1 for dets in results.values() if len(dets) > 0)
    
    print(f"\nSummary:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Images with detections: {images_with_detections}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per image: {total_detections/len(image_files):.2f}")
    
    return results

# ============================================================================
# 4. INTERACTIVE ANNOTATION UI
# ============================================================================

class AnnotationUI:
    """Interactive UI for annotating missed objects"""
    
    def __init__(self, frames_dir: Path, detections_file: Optional[Path] = None):
        """
        Initialize annotation UI
        
        Args:
            frames_dir: Directory containing frames
            detections_file: Path to detections JSON (optional)
        """
        self.frames_dir = Path(frames_dir)
        self.detections_file = detections_file or (frames_dir / "detections.json")
        
        # Load existing detections
        self.detections = {}
        if self.detections_file.exists():
            with open(self.detections_file) as f:
                self.detections = json.load(f)
        
        # Manual annotations
        self.annotations_file = frames_dir / "manual_annotations.json"
        self.manual_annotations = {}
        if self.annotations_file.exists():
            with open(self.annotations_file) as f:
                self.manual_annotations = json.load(f)
        
        # Get image files
        self.image_files = sorted(list(frames_dir.glob("*.jpg")) + 
                                 list(frames_dir.glob("*.png")))
        self.current_index = 0
        
        # Drawing state
        self.drawing = False
        self.bbox_start = None
        self.bbox_end = None
        self.current_bbox = []
        
        # Class names
        self.class_names = []
        self.load_class_names()
        
        print(f"Loaded {len(self.image_files)} images")
        print(f"Existing detections: {len(self.detections)} images")
        print(f"Manual annotations: {len(self.manual_annotations)} images")
    
    def load_class_names(self):
        """Load or create class names file"""
        class_file = self.frames_dir / "classes.txt"
        
        if class_file.exists():
            with open(class_file) as f:
                self.class_names = [line.strip() for line in f if line.strip()]
        else:
            # Default classes
            self.class_names = ["object", "person", "vehicle", "other"]
            with open(class_file, 'w') as f:
                f.write('\n'.join(self.class_names))
        
        print(f"Classes: {self.class_names}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.bbox_start = (x, y)
            self.bbox_end = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.bbox_end = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.bbox_end = (x, y)
            
            # Add bbox to current list
            if self.bbox_start and self.bbox_end:
                x1, y1 = self.bbox_start
                x2, y2 = self.bbox_end
                
                # Ensure correct order
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Only add if bbox is large enough
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.current_bbox.append([x1, y1, x2, y2])
                    print(f"Added bbox: [{x1}, {y1}, {x2}, {y2}]")
            
            self.bbox_start = None
            self.bbox_end = None
    
    def run(self):
        """Run the annotation UI"""
        
        print("\n" + "="*70)
        print("ANNOTATION UI")
        print("="*70)
        print("\nControls:")
        print("  Left click and drag: Draw bounding box")
        print("  'n': Next image")
        print("  'p': Previous image")
        print("  's': Save annotation for current image")
        print("  'u': Undo last bbox")
        print("  'c': Clear all bboxes for current image")
        print("  'q': Quit and save")
        print("  '1-9': Set class for last drawn bbox")
        print("\n")
        
        window_name = "Annotation Tool"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while self.current_index < len(self.image_files):
            image_path = self.image_files[self.current_index]
            image = cv2.imread(str(image_path))
            display_image = image.copy()
            
            # Draw existing auto-detections in blue
            filename = image_path.name
            if filename in self.detections:
                for det in self.detections[filename]:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"{det['label']}: {det['confidence']:.2f}"
                    cv2.putText(display_image, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw manual annotations in green
            if filename in self.manual_annotations:
                for ann in self.manual_annotations[filename]:
                    bbox = ann['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = ann['label']
                    cv2.putText(display_image, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw current bboxes in yellow
            for bbox in self.current_bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(display_image, "NEW", (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw current bbox being drawn
            if self.drawing and self.bbox_start and self.bbox_end:
                x1, y1 = self.bbox_start
                x2, y2 = self.bbox_end
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Add info text
            info_text = f"Image {self.current_index+1}/{len(self.image_files)} - {filename}"
            cv2.putText(display_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            bbox_count = len(self.current_bbox)
            cv2.putText(display_image, f"New bboxes: {bbox_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(window_name, display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                self.current_index = min(self.current_index + 1, len(self.image_files) - 1)
                self.current_bbox = []
            elif key == ord('p'):
                self.current_index = max(self.current_index - 1, 0)
                self.current_bbox = []
            elif key == ord('s'):
                self.save_current_annotations(filename)
                self.current_bbox = []
                self.current_index = min(self.current_index + 1, len(self.image_files) - 1)
            elif key == ord('u'):
                if self.current_bbox:
                    self.current_bbox.pop()
                    print("Undone last bbox")
            elif key == ord('c'):
                self.current_bbox = []
                print("Cleared all bboxes")
            elif ord('1') <= key <= ord('9'):
                class_idx = key - ord('1')
                if class_idx < len(self.class_names) and self.current_bbox:
                    # Set class for last bbox
                    last_bbox = self.current_bbox[-1]
                    print(f"Set class '{self.class_names[class_idx]}' for last bbox")
        
        cv2.destroyAllWindows()
        self.save_all_annotations()
        print("\nAnnotation session complete!")
    
    def save_current_annotations(self, filename: str):
        """Save annotations for current image"""
        
        if not self.current_bbox:
            print("No annotations to save")
            return
        
        # Prompt for class labels
        annotations = []
        for bbox in self.current_bbox:
            print(f"\nBBox: {bbox}")
            print("Available classes:")
            for i, class_name in enumerate(self.class_names):
                print(f"  {i+1}. {class_name}")
            
            while True:
                try:
                    choice = input(f"Select class (1-{len(self.class_names)}) or enter new class name: ")
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(self.class_names):
                            label = self.class_names[idx]
                            break
                    else:
                        label = choice
                        if label not in self.class_names:
                            self.class_names.append(label)
                        break
                except:
                    continue
            
            annotations.append({
                'bbox': bbox,
                'label': label,
                'confidence': 1.0,
                'source': 'manual'
            })
        
        self.manual_annotations[filename] = annotations
        print(f"Saved {len(annotations)} annotations for {filename}")
    
    def save_all_annotations(self):
        """Save all manual annotations to file"""
        
        with open(self.annotations_file, 'w') as f:
            json.dump(self.manual_annotations, f, indent=2)
        
        # Update classes file
        class_file = self.frames_dir / "classes.txt"
        with open(class_file, 'w') as f:
            f.write('\n'.join(self.class_names))
        
        print(f"\nSaved annotations: {self.annotations_file}")
        print(f"Total annotated images: {len(self.manual_annotations)}")

# ============================================================================
# 5. EXPORT ANNOTATED DATA
# ============================================================================

def export_annotated_data(frames_dir: Path, output_dir: Optional[Path] = None):
    """
    Export annotated data for further analysis/training
    
    Args:
        frames_dir: Directory containing frames and annotations
        output_dir: Output directory (default: frames_dir/annotated_export)
    """
    
    print("\n" + "="*70)
    print("EXPORTING ANNOTATED DATA")
    print("="*70)
    
    if output_dir is None:
        output_dir = frames_dir / "annotated_export"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    detections_file = frames_dir / "detections.json"
    annotations_file = frames_dir / "manual_annotations.json"
    metadata_file = frames_dir / "metadata.json"
    
    detections = {}
    if detections_file.exists():
        with open(detections_file) as f:
            detections = json.load(f)
    
    manual_annotations = {}
    if annotations_file.exists():
        with open(annotations_file) as f:
            manual_annotations = json.load(f)
    
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
    
    # Combine detections and manual annotations
    combined_data = {}
    
    # Get all image files
    image_files = set()
    for filename in detections.keys():
        image_files.add(filename)
    for filename in manual_annotations.keys():
        image_files.add(filename)
    
    for filename in image_files:
        combined_data[filename] = {
            'auto_detections': detections.get(filename, []),
            'manual_annotations': manual_annotations.get(filename, []),
            'all_objects': []
        }
        
        # Combine all objects
        all_objects = []
        for det in detections.get(filename, []):
            obj = det.copy()
            obj['source'] = 'auto'
            all_objects.append(obj)
        
        for ann in manual_annotations.get(filename, []):
            obj = ann.copy()
            obj['source'] = 'manual'
            all_objects.append(obj)
        
        combined_data[filename]['all_objects'] = all_objects
    
    # Save combined data
    combined_file = output_dir / "combined_annotations.json"
    with open(combined_file, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    # Export in YOLO format
    yolo_dir = output_dir / "yolo_format"
    yolo_dir.mkdir(exist_ok=True)
    (yolo_dir / "images").mkdir(exist_ok=True)
    (yolo_dir / "labels").mkdir(exist_ok=True)
    
    # Get class names
    class_file = frames_dir / "classes.txt"
    class_names = []
    if class_file.exists():
        with open(class_file) as f:
            class_names = [line.strip()]

