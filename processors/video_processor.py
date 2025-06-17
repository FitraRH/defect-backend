# processors/video_processor.py
"""
Video and real-time camera processing
"""

import cv2
import os
import time
import numpy as np
import json
from datetime import datetime
from config import *
from processors.image_processor import ImageProcessor


class VideoProcessor:
    """Handles video and camera processing"""
    
    def __init__(self, detection_core):
        self.detection_core = detection_core
        self.image_processor = ImageProcessor(detection_core)
        self.bg_subtractor = None
        self.reset_background_model()
    
    def reset_background_model(self):
        """Reset background subtractor for video processing"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False)
    
    def process_video(self, video_path, output_dir=None, save_video=True, frame_skip=None):
        """Process video file for defect detection"""
        if output_dir is None:
            output_dir = OUTPUTS_DIR / "video"
        if frame_skip is None:
            frame_skip = VIDEO_FRAME_SKIP
            
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Total frames: {total_frames}")
        
        # Initialize video writer
        video_writer = None
        if save_video:
            output_video_path = os.path.join(output_dir, f"processed_{os.path.basename(video_path)}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Reset background model
        self.reset_background_model()
        
        # Processing statistics
        stats = self._init_video_stats()
        frame_idx = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                    frame_idx += 1
                    if video_writer:
                        video_writer.write(frame)
                    continue
                
                # Process frame
                processed_frame, frame_result = self._process_video_frame(frame, output_dir, frame_idx, total_frames)
                
                if frame_result:
                    self._update_video_stats(stats, frame_result)
                
                # Save processed frame to video
                if video_writer:
                    video_writer.write(processed_frame)
                
                # Display frame (optional)
                cv2.imshow('Video Processing', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_idx += 1
                
                # Progress update
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})")
        
        except Exception as e:
            print(f"Error during video processing: {e}")
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            # Generate video summary
            return self._finalize_video_stats(stats, output_dir, video_path)
    
    def process_camera_realtime(self, camera_id=0, output_dir=None):
        """Real-time camera processing"""
        if output_dir is None:
            output_dir = OUTPUTS_DIR / "camera"
            
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Starting real-time camera processing (Camera ID: {camera_id})")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'r' to reset background model")
        print("  - Press 'c' to capture and analyze current frame")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return None
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
        
        # Reset background model and calibrate
        self.reset_background_model()
        self._calibrate_camera_background(cap)
        
        # Processing variables
        frame_count = 0
        last_analysis_time = 0
        current_status = "READY"
        last_result = None
        
        print("Real-time processing started!")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                display_frame, current_status, last_result = self._process_camera_frame(
                    frame, output_dir, last_analysis_time, current_status, last_result
                )
                
                # Show frame
                cv2.imshow('Real-time Defect Detection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Stopping real-time processing...")
                    break
                elif key == ord('s'):
                    self._save_camera_frame(frame, output_dir)
                elif key == ord('r'):
                    print("Resetting background model...")
                    self.reset_background_model()
                    current_status = "READY"
                    last_result = None
                elif key == ord('c'):
                    last_result = self._capture_and_analyze(frame, output_dir)
                    if last_result:
                        current_status = last_result['final_decision']
                
                frame_count += 1
        
        except Exception as e:
            print(f"Error in real-time processing: {e}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"Processed {frame_count} frames")
    
    def _init_video_stats(self):
        """Initialize video processing statistics"""
        return {
            'frames_processed': 0,
            'good_frames': 0,
            'defective_frames': 0,
            'defect_detections': {},
            'processing_times': [],
            'anomaly_scores': []
        }
    
    def _process_video_frame(self, frame, output_dir, frame_idx, total_frames):
        """Process a single video frame"""
        # Save frame temporarily for processing
        temp_frame_path = os.path.join(output_dir, "temp_frame.jpg")
        cv2.imwrite(temp_frame_path, frame)
        
        # Process frame
        frame_start_time = time.time()
        result = self.image_processor.process_single_image(temp_frame_path, output_dir)
        frame_processing_time = time.time() - frame_start_time
        
        # Clean up temp file
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
        
        # Annotate frame
        annotated_frame = self._annotate_frame(frame, result, frame_idx, total_frames)
        
        return annotated_frame, result
    
    def _annotate_frame(self, frame, result, frame_idx, total_frames):
        """Add annotations to video frame"""
        height = frame.shape[0]
        
        if result:
            if result['final_decision'] == 'GOOD':
                # Add green border for good products
                frame = cv2.copyMakeBorder(frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 255, 0))
                cv2.putText(frame, "GOOD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                # Add red border for defective products
                frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 255))
                cv2.putText(frame, "DEFECT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # Draw defect bounding boxes if available
                if result['defect_classification'] and 'bounding_boxes' in result['defect_classification']:
                    for defect_type, bboxes in result['defect_classification']['bounding_boxes'].items():
                        color = DEFECT_COLORS.get(
                            result['defect_classification']['class_distribution'][defect_type]['class_id'], 
                            (255, 255, 255)
                        )
                        for bbox in bboxes:
                            cv2.rectangle(frame, 
                                        (bbox['x'], bbox['y']), 
                                        (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                                        color, 2)
                            cv2.putText(frame, defect_type.upper(),
                                      (bbox['x'], bbox['y'] - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add processing info
            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (20, height - 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Score: {result['anomaly_detection']['anomaly_score']:.3f}", (20, height - 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _update_video_stats(self, stats, result):
        """Update video processing statistics"""
        stats['frames_processed'] += 1
        stats['processing_times'].append(result['processing_time'])
        stats['anomaly_scores'].append(result['anomaly_detection']['anomaly_score'])
        
        if result['final_decision'] == 'GOOD':
            stats['good_frames'] += 1
        else:
            stats['defective_frames'] += 1
            
            # Track defect types
            if 'detected_defect_types' in result:
                for defect_type in result['detected_defect_types']:
                    if defect_type not in stats['defect_detections']:
                        stats['defect_detections'][defect_type] = 0
                    stats['defect_detections'][defect_type] += 1
    
    def _finalize_video_stats(self, stats, output_dir, video_path):
        """Generate final video processing summary"""
        if stats['frames_processed'] > 0:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['avg_anomaly_score'] = np.mean(stats['anomaly_scores'])
            stats['defect_rate'] = (stats['defective_frames'] / stats['frames_processed']) * 100
            
            # Save video analysis summary
            summary_path = os.path.join(output_dir, f"video_analysis_{os.path.basename(video_path)}.json")
            with open(summary_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"Video processing complete:")
            print(f"   Frames processed: {stats['frames_processed']}")
            print(f"   Good frames: {stats['good_frames']}")
            print(f"   Defective frames: {stats['defective_frames']}")
            print(f"   Defect rate: {stats['defect_rate']:.1f}%")
            print(f"   Defect types: {list(stats['defect_detections'].keys())}")
            
            return stats
        
        return None
    
    def _calibrate_camera_background(self, cap):
        """Calibrate camera background"""
        print("Calibrating background (keep camera view clear)...")
        for i in range(30):
            ret, frame = cap.read()
            if ret:
                self.bg_subtractor.apply(frame)
                cv2.putText(frame, f"Calibrating... {i+1}/30", (20, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('Camera Calibration', frame)
                cv2.waitKey(30)
        
        cv2.destroyWindow('Camera Calibration')
    
    def _process_camera_frame(self, frame, output_dir, last_analysis_time, current_status, last_result):
        """Process single camera frame"""
        display_frame = frame.copy()
        current_time = time.time()
        
        # Periodic analysis
        if current_time - last_analysis_time > ANALYSIS_INTERVAL:
            temp_frame_path = os.path.join(output_dir, "temp_realtime_frame.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                result = self.image_processor.process_single_image(temp_frame_path, output_dir)
                if result:
                    last_result = result
                    current_status = result['final_decision']
                last_analysis_time = current_time
                
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
            except Exception as e:
                print(f"Analysis error: {e}")
        
        # Draw status overlay
        height = display_frame.shape[0]
        
        if current_status == "GOOD":
            color = (0, 255, 0)
            display_frame = cv2.copyMakeBorder(display_frame, 5, 5, 5, 5, 
                                             cv2.BORDER_CONSTANT, value=color)
        elif current_status == "DEFECT":
            color = (0, 0, 255)
            display_frame = cv2.copyMakeBorder(display_frame, 10, 10, 10, 10, 
                                             cv2.BORDER_CONSTANT, value=color)
        else:
            color = (255, 255, 0)
        
        # Add status text and info
        cv2.putText(display_frame, f"Status: {current_status}", (20, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        if last_result:
            score = last_result['anomaly_detection']['anomaly_score']
            cv2.putText(display_frame, f"Anomaly Score: {score:.3f}", (20, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if 'detected_defect_types' in last_result and last_result['detected_defect_types']:
                defects_text = ", ".join(last_result['detected_defect_types'][:3])
                cv2.putText(display_frame, f"Defects: {defects_text}", (20, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add controls info
        cv2.putText(display_frame, "Controls: Q=Quit, S=Save, R=Reset, C=Capture", 
                  (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display_frame, current_status, last_result
    
    def _save_camera_frame(self, frame, output_dir):
        """Save current camera frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"captured_frame_{timestamp}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Frame saved: {save_path}")
    
    def _capture_and_analyze(self, frame, output_dir):
        """Force capture and analysis of current frame"""
        print("Capturing and analyzing...")
        temp_frame_path = os.path.join(output_dir, "manual_capture.jpg")
        cv2.imwrite(temp_frame_path, frame)
        
        result = self.image_processor.process_single_image(temp_frame_path, output_dir)
        if result:
            print(f"Analysis result: {result['final_decision']}")
            if result['detected_defect_types']:
                print(f"Defects found: {result['detected_defect_types']}")
        
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
        
        return result