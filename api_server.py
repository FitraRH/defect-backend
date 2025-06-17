# api_server.py
"""
Flask API Server for Flutter Integration
This file replaces app.py for mobile integration
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import base64
import cv2
import numpy as np
from datetime import datetime
import tempfile
import uuid

# Import core detection system
from main import UnifiedDefectDetector, create_detector


class FlutterAPIServer:
    """API Server for Flutter integration"""
    
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        CORS(self.app)  # Allow Flutter to call API
        
        self.host = host
        self.port = port
        self.detector = None
        
        # Initialize detector
        self._initialize_detector()
        
        # Setup routes
        self._setup_routes()
    
    def _initialize_detector(self):
        """Initialize detection system"""
        try:
            self.detector = create_detector()
            if self.detector.is_ready():
                print("Detection system ready for Flutter integration")
            else:
                print("Detection system initialized but models not loaded")
        except Exception as e:
            print(f"Failed to initialize detector: {e}")
            self.detector = None
    
    def _setup_routes(self):
        """Setup API routes for Flutter"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'ok',
                'detector_ready': self.detector.is_ready() if self.detector else False,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/system-info', methods=['GET'])
        def get_system_info():
            """Get system information"""
            if not self.detector:
                return jsonify({'error': 'Detector not initialized'}), 500
            
            try:
                info = self.detector.get_system_info()
                return jsonify({
                    'status': 'success',
                    'data': info
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/detect-image', methods=['POST'])
        def detect_image():
            """Main detection endpoint for Flutter"""
            if not self.detector or not self.detector.is_ready():
                return jsonify({'error': 'Detection system not ready'}), 500
            
            try:
                # Handle different input formats
                if 'image' in request.files:
                    # File upload
                    image_file = request.files['image']
                    image_data = image_file.read()
                
                elif request.json and 'image_base64' in request.json:
                    # Base64 encoded image from Flutter
                    base64_data = request.json['image_base64']
                    if base64_data.startswith('data:image'):
                        base64_data = base64_data.split(',')[1]
                    image_data = base64.b64decode(base64_data)
                
                else:
                    return jsonify({'error': 'No image provided'}), 400
                
                # Save temporary image
                temp_id = str(uuid.uuid4())
                temp_path = f"temp_{temp_id}.jpg"
                
                with open(temp_path, 'wb') as f:
                    f.write(image_data)
                
                # Process image
                result = self.detector.process_image(temp_path)
                
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if result:
                    # Format response for Flutter
                    flutter_response = self._format_flutter_response(result)
                    return jsonify(flutter_response)
                else:
                    return jsonify({'error': 'Processing failed'}), 500
                    
            except Exception as e:
                # Clean up temp file if error occurs
                temp_path = f"temp_{temp_id}.jpg"
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/batch-detect', methods=['POST'])
        def batch_detect():
            """Batch detection endpoint (for multiple images from Flutter)"""
            if not self.detector or not self.detector.is_ready():
                return jsonify({'error': 'Detection system not ready'}), 500
            
            try:
                if not request.json or 'images' not in request.json:
                    return jsonify({'error': 'No images provided'}), 400
                
                images_data = request.json.get('images', [])
                if not images_data:
                    return jsonify({'error': 'No images provided'}), 400
                
                results = []
                
                for idx, image_b64 in enumerate(images_data):
                    try:
                        # Decode base64 image
                        if image_b64.startswith('data:image'):
                            image_b64 = image_b64.split(',')[1]
                        image_data = base64.b64decode(image_b64)
                        
                        # Save temporary image
                        temp_path = f"temp_batch_{idx}_{uuid.uuid4()}.jpg"
                        with open(temp_path, 'wb') as f:
                            f.write(image_data)
                        
                        # Process image
                        result = self.detector.process_image(temp_path)
                        
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        if result:
                            flutter_result = self._format_flutter_response(result)
                            flutter_result['image_index'] = idx
                            results.append(flutter_result)
                    
                    except Exception as e:
                        # Clean up and continue with next image
                        temp_path = f"temp_batch_{idx}_{uuid.uuid4()}.jpg"
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        print(f"Error processing image {idx}: {e}")
                        continue
                
                # Generate batch summary
                total_images = len(results)
                good_products = sum(1 for r in results if r['final_decision'] == 'GOOD')
                defective_products = total_images - good_products
                
                return jsonify({
                    'status': 'success',
                    'summary': {
                        'total_images': total_images,
                        'good_products': good_products,
                        'defective_products': defective_products,
                        'defect_rate': (defective_products / total_images * 100) if total_images > 0 else 0
                    },
                    'results': results
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/update-thresholds', methods=['POST'])
        def update_thresholds():
            """Update detection thresholds from Flutter"""
            if not self.detector:
                return jsonify({'error': 'Detector not initialized'}), 500
            
            try:
                if not request.json:
                    return jsonify({'error': 'No data provided'}), 400
                
                data = request.json
                anomaly_threshold = data.get('anomaly_threshold')
                defect_threshold = data.get('defect_threshold')
                
                self.detector.update_thresholds(anomaly_threshold, defect_threshold)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Thresholds updated successfully'
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/get-visualization/<result_id>', methods=['GET'])
        def get_visualization(result_id):
            """Get visualization image for Flutter display"""
            try:
                viz_path = f"outputs/analysis_{result_id}.png"
                if os.path.exists(viz_path):
                    return send_file(viz_path, mimetype='image/png')
                else:
                    return jsonify({'error': 'Visualization not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _format_flutter_response(self, result):
        """Format detection result for Flutter consumption"""
        try:
            # Clean response format for Flutter
            flutter_response = {
                'status': 'success',
                'final_decision': result['final_decision'],
                'processing_time': round(result['processing_time'], 2),
                'timestamp': result['timestamp'],
                
                # Anomaly detection info
                'anomaly_detection': {
                    'anomaly_score': round(result['anomaly_detection']['anomaly_score'], 4),
                    'decision': result['anomaly_detection']['decision'],
                    'threshold_used': result['anomaly_detection']['threshold_used']
                },
                
                # Defect classification (if available)
                'defect_classification': None,
                'detected_defects': result.get('detected_defect_types', []),
                'defect_count': len(result.get('detected_defect_types', [])),
                
                # Simplified bounding boxes for Flutter
                'bounding_boxes': []
            }
            
            # Add defect classification info if available
            if result.get('defect_classification'):
                defect_result = result['defect_classification']
                
                flutter_response['defect_classification'] = {
                    'detected_defects': defect_result['detected_defects'],
                    'total_defect_types': len(defect_result['detected_defects'])
                }
                
                # Simplify bounding boxes for Flutter
                for defect_type, bboxes in defect_result.get('bounding_boxes', {}).items():
                    for bbox in bboxes:
                        flutter_response['bounding_boxes'].append({
                            'defect_type': defect_type,
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height'],
                            'center_x': bbox['center_x'],
                            'center_y': bbox['center_y'],
                            'area': bbox['area']
                        })
            
            return flutter_response
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'final_decision': 'ERROR'
            }
    
    def run(self, debug=False):
        """Start the API server"""
        print(f"Starting Flutter API Server on {self.host}:{self.port}")
        print(f"Flutter can connect to: http://{self.host}:{self.port}")
        print(f"Health check: http://{self.host}:{self.port}/api/health")
        print(f"Main endpoint: http://{self.host}:{self.port}/api/detect-image")
        
        self.app.run(host=self.host, port=self.port, debug=debug)


# Convenience function
def create_flutter_api(host='0.0.0.0', port=5000):
    """Create and return Flask API server for Flutter"""
    return FlutterAPIServer(host=host, port=port)


if __name__ == "__main__":
    # Start API server for Flutter
    api_server = create_flutter_api()
    api_server.run(debug=True)