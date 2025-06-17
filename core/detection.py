# core/detection.py
"""
Core detection logic for anomaly detection and defect classification
"""

import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import *


class DetectionCore:
    """Core detection functionality"""
    
    def __init__(self, anomalib_model, hrnet_model, device='cuda'):
        self.anomalib_model = anomalib_model
        self.hrnet_model = hrnet_model
        self.device = device
        
        # Preprocessing for HRNet
        self.hrnet_transform = A.Compose([
            A.Resize(*IMAGE_SIZE),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ToTensorV2()
        ])
    
    def detect_anomaly(self, image_path):
        """
        Step 1: Use Anomalib to detect if product is good or defective
        
        Returns:
            dict: Contains anomaly detection results
        """
        if not self.anomalib_model:
            raise ValueError("Anomalib model not loaded")
        
        try:
            # Run Anomalib inference
            result = self.anomalib_model.predict(image=image_path)
            
            # Process anomaly results
            if isinstance(result.pred_score, torch.Tensor):
                anomaly_score = float(result.pred_score.cpu().item())
            else:
                anomaly_score = float(result.pred_score)
                
            if isinstance(result.pred_label, torch.Tensor):
                is_anomalous = bool(result.pred_label.cpu().item())
            else:
                is_anomalous = bool(result.pred_label)
            
            # Get anomaly mask if available
            anomaly_mask = None
            if hasattr(result, 'pred_mask') and result.pred_mask is not None:
                if isinstance(result.pred_mask, torch.Tensor):
                    anomaly_mask = result.pred_mask.cpu().numpy()
                else:
                    anomaly_mask = result.pred_mask
                
                if len(anomaly_mask.shape) > 2:
                    anomaly_mask = anomaly_mask[0]
            
            return {
                'is_anomalous': is_anomalous,
                'anomaly_score': anomaly_score,
                'anomaly_mask': anomaly_mask,
                'threshold_used': ANOMALY_THRESHOLD,
                'decision': 'DEFECT' if (is_anomalous and anomaly_score > ANOMALY_THRESHOLD) else 'GOOD'
            }
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return None
    
    def classify_defects(self, image_path, region_mask=None):
        """
        Step 2: Use HRNet to classify specific defect types
        
        Args:
            image_path: Path to image
            region_mask: Optional mask to focus on specific regions
            
        Returns:
            dict: Contains defect classification results
        """
        if not self.hrnet_model:
            raise ValueError("HRNet model not loaded")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image_rgb.shape[:2]
            
            # Apply region mask if provided
            if region_mask is not None:
                # Resize mask to match image
                region_mask = cv2.resize(region_mask, (original_size[1], original_size[0]))
                # Apply mask to image
                masked_image = image_rgb.copy()
                masked_image[region_mask < 0.5] = 0
                image_rgb = masked_image
            
            # Preprocess for HRNet
            transformed = self.hrnet_transform(image=image_rgb)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # HRNet inference
            with torch.no_grad():
                output = self.hrnet_model(input_tensor)
                predictions = torch.softmax(output, dim=1)
                predicted_mask = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()
                confidence_scores = torch.max(predictions, dim=1)[0].squeeze().cpu().numpy()
            
            # Resize predictions to original size
            predicted_mask = cv2.resize(predicted_mask.astype(np.uint8), 
                                      (original_size[1], original_size[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            confidence_scores = cv2.resize(confidence_scores, 
                                         (original_size[1], original_size[0]), 
                                         interpolation=cv2.INTER_LINEAR)
            
            # Analyze defect predictions
            defect_analysis = self._analyze_defect_predictions(predicted_mask, confidence_scores)
            
            return {
                'predicted_mask': predicted_mask,
                'confidence_scores': confidence_scores,
                'defect_analysis': defect_analysis,
                'detected_defects': defect_analysis['detected_defects'],
                'bounding_boxes': defect_analysis['bounding_boxes'],
                'class_distribution': defect_analysis['class_distribution']
            }
            
        except Exception as e:
            print(f"Error in defect classification: {e}")
            return None
    
    def _analyze_defect_predictions(self, predicted_mask, confidence_scores):
        """Analyze HRNet predictions to extract defect information"""
        h, w = predicted_mask.shape
        total_pixels = h * w
        
        analysis = {
            'detected_defects': [],
            'class_distribution': {},
            'bounding_boxes': {},
            'defect_statistics': {}
        }
        
        # Analyze each defect class
        for class_id, class_name in SPECIFIC_DEFECT_CLASSES.items():
            class_mask = (predicted_mask == class_id)
            pixel_count = np.sum(class_mask)
            percentage = (pixel_count / total_pixels) * 100
            
            analysis['class_distribution'][class_name] = {
                'pixel_count': int(pixel_count),
                'percentage': percentage,
                'class_id': class_id
            }
            
            # Only process actual defects (not background)
            if class_id > 0 and pixel_count > 0:
                # Apply confidence threshold
                confident_mask = class_mask & (confidence_scores > DEFECT_CONFIDENCE_THRESHOLD)
                confident_pixels = np.sum(confident_mask)
                
                # Detection criteria
                min_pixels = max(MIN_DEFECT_PIXELS, total_pixels * MIN_DEFECT_PERCENTAGE)
                
                if confident_pixels > min_pixels or pixel_count > total_pixels * 0.1:
                    analysis['detected_defects'].append(class_name)
                    
                    # Extract bounding boxes
                    bboxes = self._extract_bounding_boxes(
                        confident_mask if confident_pixels > min_pixels else class_mask
                    )
                    analysis['bounding_boxes'][class_name] = bboxes
                    
                    # Calculate statistics
                    analysis['defect_statistics'][class_name] = {
                        'confident_pixels': int(confident_pixels if confident_pixels > 0 else pixel_count),
                        'confidence_ratio': confident_pixels / pixel_count if pixel_count > 0 else 0,
                        'avg_confidence': float(np.mean(confidence_scores[
                            confident_mask if confident_pixels > 0 else class_mask
                        ])),
                        'max_confidence': float(np.max(confidence_scores[
                            confident_mask if confident_pixels > 0 else class_mask
                        ])),
                        'num_regions': len(bboxes)
                    }
        
        return analysis
    
    def _extract_bounding_boxes(self, mask):
        """Extract bounding boxes from binary mask"""
        try:
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bounding_boxes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= MIN_BBOX_AREA:
                    x, y, w, h = cv2.boundingRect(contour)
                    bounding_boxes.append({
                        'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h),
                        'area': int(area),
                        'center_x': int(x + w/2), 'center_y': int(y + h/2)
                    })
            
            return bounding_boxes
        except:
            return []