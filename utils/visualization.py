# utils/visualization.py
"""
Visualization utilities for analysis results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from config import DEFECT_COLORS, SPECIFIC_DEFECT_CLASSES


def create_visualization(result, output_dir):
    """Create comprehensive visualization of analysis results"""
    try:
        # Load original image
        image = cv2.imread(result['image_path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Unified Defect Detection Results - {os.path.basename(result["image_path"])}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Product Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Plot 2: Anomaly detection result
        _plot_anomaly_detection(axes[0, 1], image_rgb, result['anomaly_detection'])
        
        # Plot 3: Defect classification (if available)
        _plot_defect_classification(axes[1, 0], result)
        
        # Plot 4: Combined result with bounding boxes
        _plot_combined_result(axes[1, 1], image_rgb, result)
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_filename = f"analysis_{timestamp}_{os.path.splitext(os.path.basename(result['image_path']))[0]}.png"
        viz_path = os.path.join(output_dir, viz_filename)
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None


def _plot_anomaly_detection(ax, image_rgb, anomaly_result):
    """Plot anomaly detection results"""
    if anomaly_result['anomaly_mask'] is not None:
        # Show anomaly mask
        mask = anomaly_result['anomaly_mask']
        if len(mask.shape) > 2:
            mask = mask[0]
        mask_resized = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
        
        # Create heatmap
        heatmap = plt.cm.hot(mask_resized)
        ax.imshow(image_rgb)
        ax.imshow(heatmap, alpha=0.6)
        ax.set_title(f'Anomaly Detection - {anomaly_result["decision"]} \n'
                    f'Score: {anomaly_result["anomaly_score"]:.3f}', fontweight='bold')
    else:
        # Show decision with colored overlay
        overlay = image_rgb.copy()
        if anomaly_result['decision'] == 'GOOD':
            overlay[:, :, 1] = np.minimum(overlay[:, :, 1] + 30, 255)  # Green tint
        else:
            overlay[:, :, 0] = np.minimum(overlay[:, :, 0] + 30, 255)  # Red tint
        
        ax.imshow(overlay)
        ax.set_title(f'Anomaly Detection - {anomaly_result["decision"]} \n'
                    f'Score: {anomaly_result["anomaly_score"]:.3f}', fontweight='bold')
    ax.axis('off')


def _plot_defect_classification(ax, result):
    """Plot defect classification results"""
    if result['defect_classification']:
        defect_result = result['defect_classification']
        predicted_mask = defect_result['predicted_mask']
        
        # Create colored defect mask
        colored_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
        for class_id, color in DEFECT_COLORS.items():
            class_pixels = (predicted_mask == class_id)
            colored_mask[class_pixels] = color
        
        ax.imshow(colored_mask)
        ax.set_title(f'Defect Classification \n'
                    f'Types: {", ".join(defect_result["detected_defects"])}', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Defect Classification\n(Product is GOOD)', 
              ha='center', va='center', transform=ax.transAxes,
              fontsize=14, color='green', fontweight='bold')
        ax.set_title('Defect Classification', fontweight='bold')
    ax.axis('off')


def _plot_combined_result(ax, image_rgb, result):
    """Plot combined result with bounding boxes"""
    result_image = image_rgb.copy()
    
    # Add anomaly bounding box (whole product)
    if result['anomaly_detection']['decision'] == 'DEFECT':
        h, w = result_image.shape[:2]
        cv2.rectangle(result_image, (10, 10), (w-10, h-10), (255, 0, 0), 5)  # Red border for defect
        
        # Add defect-specific bounding boxes
        if result['defect_classification'] and 'bounding_boxes' in result['defect_classification']:
            for defect_type, bboxes in result['defect_classification']['bounding_boxes'].items():
                class_id = result['defect_classification']['class_distribution'][defect_type]['class_id']
                color = DEFECT_COLORS.get(class_id, (255, 255, 255))
                
                for bbox in bboxes:
                    cv2.rectangle(result_image, 
                                (bbox['x'], bbox['y']), 
                                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                                color, 3)
                    cv2.putText(result_image, defect_type.upper(),
                              (bbox['x'], bbox['y'] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        h, w = result_image.shape[:2]
        cv2.rectangle(result_image, (10, 10), (w-10, h-10), (0, 255, 0), 5)  # Green border for good
    
    ax.imshow(result_image)
    ax.set_title(f'Final Result: {result["final_decision"]} \n'
                f'Processing time: {result["processing_time"]:.2f}s', fontweight='bold')
    ax.axis('off')


def create_defect_legend():
    """Create a legend for defect colors"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    legend_elements = []
    for class_id, class_name in SPECIFIC_DEFECT_CLASSES.items():
        if class_id > 0:  # Skip background
            color = [c/255.0 for c in DEFECT_COLORS[class_id]]  # Normalize to 0-1
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                               label=class_name.replace('_', ' ').title()))
    
    ax.legend(handles=legend_elements, loc='center', fontsize=12)
    ax.set_title('Defect Type Color Legend', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    return fig


def save_batch_visualization(batch_results, output_dir):
    """Create summary visualization for batch processing"""
    try:
        summary = batch_results['summary']
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Batch Processing Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Good vs Defective distribution
        labels = ['Good Products', 'Defective Products']
        sizes = [summary['good_products'], summary['defective_products']]
        colors = ['#90EE90', '#FFB6C1']
        
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Product Quality Distribution')
        
        # Plot 2: Defect types frequency
        if summary['defect_types_found']:
            defect_counts = {}
            for result in batch_results['results']:
                if 'detected_defect_types' in result:
                    for defect in result['detected_defect_types']:
                        defect_counts[defect] = defect_counts.get(defect, 0) + 1
            
            defect_names = list(defect_counts.keys())
            defect_frequencies = list(defect_counts.values())
            
            axes[0, 1].bar(defect_names, defect_frequencies)
            axes[0, 1].set_title('Defect Type Frequency')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Defects Found', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=14)
            axes[0, 1].set_title('Defect Type Frequency')
        
        # Plot 3: Processing time distribution
        processing_times = summary['processing_times']
        axes[1, 0].hist(processing_times, bins=20, alpha=0.7, color='skyblue')
        axes[1, 0].set_title('Processing Time Distribution')
        axes[1, 0].set_xlabel('Processing Time (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Summary statistics
        stats_text = f"""
        Total Images: {summary['total_images']}
        Good Products: {summary['good_products']}
        Defective Products: {summary['defective_products']}
        Success Rate: {((summary['total_images'] - summary['failed_processing']) / summary['total_images'] * 100):.1f}%
        Avg Processing Time: {summary['avg_processing_time']:.2f}s
        Defect Types Found: {len(summary['defect_types_found'])}
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save batch summary visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_viz_path = os.path.join(output_dir, f"batch_summary_{timestamp}.png")
        plt.savefig(summary_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return summary_viz_path
        
    except Exception as e:
        print(f"Error creating batch visualization: {e}")
        return None