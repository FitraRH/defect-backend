# processors/image_processor.py
"""
Image processing functions for single and batch image handling
"""

import os
import time
import glob
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from config import *
from utils.visualization import create_visualization
from utils.reports import save_analysis_report


class ImageProcessor:
    """Handles image processing workflows"""
    
    def __init__(self, detection_core):
        self.detection_core = detection_core
    
    def process_single_image(self, image_path, output_dir=None):
        """
        Complete workflow: Anomaly detection -> Defect classification
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            
        Returns:
            dict: Complete analysis results
        """
        if output_dir is None:
            output_dir = OUTPUTS_DIR
            
        start_time = time.time()
        
        print(f"Processing image: {os.path.basename(image_path)}")
        
        try:
            # Step 1: Anomaly Detection
            print("Step 1: Anomaly detection...")
            anomaly_result = self.detection_core.detect_anomaly(image_path)
            
            if not anomaly_result:
                return None
            
            result = {
                'image_path': image_path,
                'timestamp': datetime.now().isoformat(),
                'anomaly_detection': anomaly_result,
                'defect_classification': None,
                'processing_time': 0,
                'final_decision': anomaly_result['decision']
            }
            
            # Step 2: If defect detected, classify defect types
            if anomaly_result['decision'] == 'DEFECT':
                print("Step 2: Defect classification...")
                defect_result = self.detection_core.classify_defects(image_path, anomaly_result['anomaly_mask'])
                
                if defect_result:
                    result['defect_classification'] = defect_result
                    result['detected_defect_types'] = defect_result['detected_defects']
            else:
                print("Product classified as GOOD - no defect classification needed")
                result['detected_defect_types'] = []
            
            # Calculate total processing time
            result['processing_time'] = time.time() - start_time
            
            # Save visualization
            visualization_path = create_visualization(result, output_dir)
            result['visualization_path'] = visualization_path
            
            # Save analysis report
            report_path = save_analysis_report(result, output_dir)
            result['report_path'] = report_path
            
            print(f"Processing complete in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def process_batch_images(self, input_folder, output_folder=None):
        """Process all images in a folder"""
        if output_folder is None:
            output_folder = OUTPUTS_DIR / "batch"
            
        os.makedirs(output_folder, exist_ok=True)
        
        # Find image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(input_folder, '**', ext), recursive=True))
        
        if not image_files:
            print(f"No image files found in {input_folder}")
            return None
        
        print(f"Processing {len(image_files)} images...")
        
        results = []
        summary = {
            'total_images': len(image_files),
            'good_products': 0,
            'defective_products': 0,
            'defect_types_found': set(),
            'processing_times': [],
            'failed_processing': 0
        }
        
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            result = self.process_single_image(image_path, output_folder)
            
            if result:
                results.append(result)
                summary['processing_times'].append(result['processing_time'])
                
                if result['final_decision'] == 'GOOD':
                    summary['good_products'] += 1
                else:
                    summary['defective_products'] += 1
                    if 'detected_defect_types' in result:
                        summary['defect_types_found'].update(result['detected_defect_types'])
            else:
                summary['failed_processing'] += 1
        
        # Save batch summary
        summary['defect_types_found'] = list(summary['defect_types_found'])
        summary['avg_processing_time'] = np.mean(summary['processing_times']) if summary['processing_times'] else 0
        
        summary_path = os.path.join(output_folder, 'batch_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Batch processing complete:")
        print(f"   Good products: {summary['good_products']}")
        print(f"   Defective products: {summary['defective_products']}")
        print(f"   Defect types found: {summary['defect_types_found']}")
        print(f"   Results saved to: {output_folder}")
        
        return {'results': results, 'summary': summary}