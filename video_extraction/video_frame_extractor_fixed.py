# save as: video_frame_extractor_fixed.py

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
import sys

class VideoFrameExtractor:
    """Extract frames from video at specified rate with support for various formats"""
    
    def __init__(self, video_path: str, output_base_dir: str = "extracted_frames", 
                 use_ffmpeg: bool = False, deinterlace: bool = True):
        """
        Initialize video frame extractor
        
        Args:
            video_path: Path to video file
            output_base_dir: Base directory for output
            use_ffmpeg: Use FFmpeg directly instead of OpenCV (better for problematic formats)
            deinterlace: Apply deinterlacing filter for interlaced videos
        """
        self.video_path = video_path
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.use_ffmpeg = use_ffmpeg
        self.deinterlace = deinterlace
        
        # Validate video
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check if FFmpeg is available
        self.ffmpeg_available = self._check_ffmpeg()
        
        # Auto-switch to FFmpeg for problematic formats
        video_ext = Path(video_path).suffix.lower()
        problematic_formats = ['.mpg', '.mpeg', '.vob', '.ts', '.m2ts', '.mts']
        if video_ext in problematic_formats and self.ffmpeg_available:
            print(f"Detected {video_ext} format - using FFmpeg for better compatibility")
            self.use_ffmpeg = True
        
        # Get video properties
        self._get_video_properties()
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _get_video_properties(self):
        """Get video properties using OpenCV or FFprobe"""
        
        if self.use_ffmpeg and self.ffmpeg_available:
            self._get_properties_ffmpeg()
        else:
            self._get_properties_opencv()
    
    def _get_properties_opencv(self):
        """Get properties using OpenCV"""
        try:
            # Try different backends
            backends = [
                cv2.CAP_FFMPEG,
                cv2.CAP_ANY,
                cv2.CAP_GSTREAMER,
            ]
            
            self.cap = None
            for backend in backends:
                cap = cv2.VideoCapture(self.video_path, backend)
                if cap.isOpened():
                    self.cap = cap
                    print(f"Opened video with backend: {backend}")
                    break
            
            if self.cap is None or not self.cap.isOpened():
                raise ValueError(f"Cannot open video with OpenCV: {self.video_path}")
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            
            # Handle invalid FPS
            if self.fps <= 0 or self.fps > 1000:
                print(f"Warning: Invalid FPS {self.fps}, attempting to detect...")
                self.fps = self._detect_fps_opencv()
            
            # Handle invalid frame count
            if self.total_frames <= 0:
                print("Warning: Invalid frame count, will count during extraction")
                self.total_frames = None
            
        except Exception as e:
            print(f"OpenCV failed: {e}")
            if self.ffmpeg_available:
                print("Falling back to FFmpeg...")
                self.use_ffmpeg = True
                self._get_properties_ffmpeg()
            else:
                raise ValueError(f"Cannot open video and FFmpeg not available: {e}")
        
        self._print_video_info()
    
    def _detect_fps_opencv(self) -> float:
        """Detect FPS by reading timestamps of frames"""
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            timestamps = []
            for i in range(min(100, self.total_frames or 100)):
                ret, _ = self.cap.read()
                if not ret:
                    break
                timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamps.append(timestamp)
            
            if len(timestamps) > 1:
                diffs = np.diff(timestamps)
                avg_diff = np.median(diffs)
                fps = 1000.0 / avg_diff if avg_diff > 0 else 25.0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return fps
        except:
            pass
        
        return 25.0  # Default fallback
    
    def _get_properties_ffmpeg(self):
        """Get properties using FFprobe"""
        try:
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
                raise ValueError(f"FFprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
            stream = data['streams'][0]
            
            self.width = int(stream['width'])
            self.height = int(stream['height'])
            
            # Parse frame rate
            fps_str = stream['r_frame_rate']
            num, den = map(int, fps_str.split('/'))
            self.fps = num / den if den != 0 else 25.0
            
            # Get frame count
            if 'nb_frames' in stream:
                self.total_frames = int(stream['nb_frames'])
            else:
                # Estimate from duration
                duration = float(stream.get('duration', 0))
                self.total_frames = int(duration * self.fps) if duration > 0 else None
            
            self.duration = self.total_frames / self.fps if self.total_frames and self.fps > 0 else 0
            
        except Exception as e:
            raise ValueError(f"Failed to get video properties with FFprobe: {e}")
        
        self._print_video_info()
    
    def _print_video_info(self):
        """Print video information"""
        print(f"\nVideo Info:")
        print(f"  Path: {self.video_path}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Total Frames: {self.total_frames if self.total_frames else 'Unknown'}")
        print(f"  Duration: {self.duration:.2f}s" if self.duration else "  Duration: Unknown")
        print(f"  Method: {'FFmpeg' if self.use_ffmpeg else 'OpenCV'}")
    
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
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = Path(self.video_path).stem
            random_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            output_dir = self.output_base_dir / f"{video_name}_{timestamp}_{random_hash}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Choose extraction method
        if self.use_ffmpeg and self.ffmpeg_available:
            return self._extract_frames_ffmpeg(
                output_dir, frames_per_second, start_time, end_time, max_frames
            )
        else:
            return self._extract_frames_opencv(
                output_dir, frames_per_second, start_time, end_time, max_frames
            )
    
    def _extract_frames_opencv(self, output_dir: Path, frames_per_second: float,
                               start_time: float, end_time: Optional[float],
                               max_frames: Optional[int]) -> Tuple[Path, dict]:
        """Extract frames using OpenCV"""
        
        # Calculate frame interval
        frame_interval = int(self.fps / frames_per_second)
        if frame_interval < 1:
            frame_interval = 1
            frames_per_second = self.fps
            print(f"Warning: Adjusted to {frames_per_second:.2f} fps (max for this video)")
        
        # Calculate start and end frames
        start_frame = int(start_time * self.fps)
        if end_time is None:
            end_frame = self.total_frames if self.total_frames else float('inf')
        else:
            end_frame = int(end_time * self.fps)
        
        # Set video position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Metadata
        metadata = {
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
            'extraction_method': 'opencv',
            'frames': []
        }
        
        # Extract frames
        frame_count = 0
        saved_count = 0
        
        total_to_extract = max_frames if max_frames else "unknown"
        print(f"\nExtracting frames to: {output_dir}")
        print(f"Frame rate: {frames_per_second:.2f} fps")
        print(f"Estimated frames to extract: {total_to_extract}")
        
        pbar = tqdm(desc="Extracting frames")
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            if max_frames and saved_count >= max_frames:
                break
            
            current_frame_num = start_frame + frame_count
            if end_frame != float('inf') and current_frame_num >= end_frame:
                break
            
            # Save frame at interval
            if frame_count % frame_interval == 0:
                # Apply deinterlacing if needed
                if self.deinterlace:
                    frame = self._deinterlace_frame(frame)
                
                # Generate filename
                timestamp_sec = current_frame_num / self.fps
                filename = f"frame_{saved_count:06d}_t{timestamp_sec:.2f}s.jpg"
                filepath = output_dir / filename
                
                # Save frame
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
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
            
            frame_count += 1
        
        pbar.close()
        
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
    
    def _extract_frames_ffmpeg(self, output_dir: Path, frames_per_second: float,
                               start_time: float, end_time: Optional[float],
                               max_frames: Optional[int]) -> Tuple[Path, dict]:
        """Extract frames using FFmpeg (more reliable for problematic formats)"""
        
        print(f"\nExtracting frames with FFmpeg to: {output_dir}")
        print(f"Frame rate: {frames_per_second:.2f} fps")
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-i', self.video_path]
        
        # Add start time
        if start_time > 0:
            cmd.extend(['-ss', str(start_time)])
        
        # Add end time
        if end_time:
            duration = end_time - start_time
            cmd.extend(['-t', str(duration)])
        
        # Add deinterlacing filter if needed
        vf_filters = []
        if self.deinterlace:
            vf_filters.append('yadif=0:-1:0')  # Deinterlace filter
        
        # Add framerate filter
        vf_filters.append(f'fps={frames_per_second}')
        
        if vf_filters:
            cmd.extend(['-vf', ','.join(vf_filters)])
        
        # Output settings
        output_pattern = str(output_dir / 'frame_%06d.jpg')
        cmd.extend([
            '-qscale:v', '2',  # High quality JPEG
            '-frame_pts', '1',  # Include timestamp info
        ])
        
        # Add frame limit if specified
        if max_frames:
            cmd.extend(['-frames:v', str(max_frames)])
        
        cmd.append(output_pattern)
        
        # Run FFmpeg
        print(f"\nRunning: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            
            # Monitor progress
            for line in process.stderr:
                if 'frame=' in line:
                    # Extract frame number from FFmpeg output
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'frame=':
                                frame_num = parts[i+1]
                                print(f"\rExtracting frame {frame_num}", end='', flush=True)
                    except:
                        pass
            
            process.wait()
            print()  # New line after progress
            
            if process.returncode != 0:
                stderr = process.stderr.read() if process.stderr else ""
                raise RuntimeError(f"FFmpeg failed with return code {process.returncode}\n{stderr}")
            
        except Exception as e:
            raise RuntimeError(f"FFmpeg extraction failed: {e}")
        
        # Get list of extracted frames
        frame_files = sorted(output_dir.glob('frame_*.jpg'))
        
        # Rename files to include timestamp and create metadata
        metadata = {
            'video_path': self.video_path,
            'video_name': Path(self.video_path).name,
            'output_dir': str(output_dir),
            'extraction_time': datetime.now().isoformat(),
            'frames_per_second': frames_per_second,
            'video_fps': self.fps,
            'start_time': start_time,
            'end_time': end_time if end_time else self.duration,
            'video_resolution': f"{self.width}x{self.height}",
            'extraction_method': 'ffmpeg',
            'deinterlaced': self.deinterlace,
            'frames': []
        }
        
        print(f"\nRenaming and cataloging {len(frame_files)} frames...")
        
        for i, old_path in enumerate(tqdm(frame_files, desc="Processing frames")):
            timestamp_sec = start_time + (i / frames_per_second)
            new_filename = f"frame_{i:06d}_t{timestamp_sec:.2f}s.jpg"
            new_path = output_dir / new_filename
            
            old_path.rename(new_path)
            
            frame_metadata = {
                'filename': new_filename,
                'filepath': str(new_path),
                'frame_number': int(timestamp_sec * self.fps),
                'timestamp': timestamp_sec,
                'extracted_index': i
            }
            metadata['frames'].append(frame_metadata)
        
        metadata['total_frames_extracted'] = len(frame_files)
        
        # Save metadata
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nExtraction complete!")
        print(f"Frames saved: {len(frame_files)}")
        print(f"Output directory: {output_dir}")
        print(f"Metadata saved: {metadata_path}")
        
        return output_dir, metadata
    
    def _deinterlace_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Simple deinterlacing by blending fields
        For better results, use FFmpeg's yadif filter
        """
        try:
            # Create deinterlaced frame by averaging adjacent lines
            deinterlaced = frame.copy()
            deinterlaced[1::2] = (frame[0:-1:2].astype(float) + frame[2::2].astype(float)) / 2
            return deinterlaced.astype(np.uint8)
        except:
            return frame
    
    def __del__(self):
        """Release video capture"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


# ============================================================================
# HELPER FUNCTION: Test Video Compatibility
# ============================================================================

def test_video_compatibility(video_path: str) -> dict:
    """
    Test if a video can be opened and report compatibility
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with compatibility information
    """
    
    print(f"Testing video: {video_path}")
    print("="*70)
    
    results = {
        'path': video_path,
        'opencv_compatible': False,
        'ffmpeg_available': False,
        'properties': {},
        'recommendations': []
    }
    
    # Test OpenCV
    print("\nTesting OpenCV...")
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            results['opencv_compatible'] = True
            results['properties']['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            results['properties']['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            results['properties']['fps'] = cap.get(cv2.CAP_PROP_FPS)
            results['properties']['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print("✓ OpenCV can read video")
            else:
                print("⚠ OpenCV opened video but cannot read frames")
                results['opencv_compatible'] = False
                results['recommendations'].append("Use FFmpeg for extraction")
            
            cap.release()
        else:
            print("✗ OpenCV cannot open video")
            results['recommendations'].append("Use FFmpeg for extraction")
    except Exception as e:
        print(f"✗ OpenCV error: {e}")
        results['recommendations'].append("Use FFmpeg for extraction")
    
    # Test FFmpeg
    print("\nTesting FFmpeg...")
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            results['ffmpeg_available'] = True
            print("✓ FFmpeg is available")
        else:
            print("✗ FFmpeg is not working")
    except FileNotFoundError:
        print("✗ FFmpeg is not installed")
        results['recommendations'].append("Install FFmpeg: https://ffmpeg.org/download.html")
    except Exception as e:
        print(f"✗ FFmpeg error: {e}")
    
    # Test FFprobe
    if results['ffmpeg_available']:
        print("\nGetting detailed info with FFprobe...")
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name,width,height,r_frame_rate,pix_fmt',
                '-of', 'json',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                stream = data['streams'][0]
                results['properties'].update({
                    'codec': stream.get('codec_name'),
                    'pixel_format': stream.get('pix_fmt')
                })
                print(f"  Codec: {stream.get('codec_name')}")
                print(f"  Pixel Format: {stream.get('pix_fmt')}")
                
                # Check for interlaced video
                pix_fmt = stream.get('pix_fmt', '')
                if 'yuv420p' in pix_fmt:
                    results['recommendations'].append("Video may be interlaced - use deinterlacing")
        except Exception as e:
            print(f"  FFprobe error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results['opencv_compatible']:
        print("✓ Video is compatible with OpenCV")
    elif results['ffmpeg_available']:
        print("✓ Video can be processed with FFmpeg")
    else:
        print("✗ Video cannot be processed - install FFmpeg")
    
    if results['recommendations']:
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"  • {rec}")
    
    return results


# ============================================================================
# FFmpeg Installation Helper
# ============================================================================

def print_ffmpeg_installation_guide():
    """Print instructions for installing FFmpeg"""
    
    print("\n" + "="*70)
    print("FFMPEG INSTALLATION GUIDE")
    print("="*70)
    
    print("\n📦 Ubuntu/Debian:")
    print("  sudo apt update")
    print("  sudo apt install ffmpeg")
    
    print("\n📦 macOS:")
    print("  brew install ffmpeg")
    
    print("\n📦 Windows:")
    print("  1. Download from: https://ffmpeg.org/download.html")
    print("  2. Extract to C:\\ffmpeg")
    print("  3. Add C:\\ffmpeg\\bin to PATH")
    print("  ")
    print("  Or use Chocolatey:")
    print("  choco install ffmpeg")
    
    print("\n📦 Using conda:")
    print("  conda install -c conda-forge ffmpeg")
    
    print("\n" + "="*70)


# Export the fixed class
__all__ = ['VideoFrameExtractor', 'test_video_compatibility', 'print_ffmpeg_installation_guide']