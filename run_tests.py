# run_tests.py
"""
Main Test Runner - Test all backend functionality
Run: python run_tests.py
"""

import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from main import UnifiedDefectDetector, create_detector, quick_detect


class BackendTester:
    """Comprehensive backend testing class"""
    
    def __init__(self):
        self.detector = None
        self.test_results = {}
        self.test_images = []
        self.start_time = time.time()
        
        # Create test output directory
        self.test_output_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        print("Backend Functionality Tester Initialized")
        print(f"Test output directory: {self.test_output_dir}")
    
    def setup_test_environment(self):
        """Setup test environment and create test data"""
        print("\nSetting up test environment...")
        
        # Create test images directory
        test_data_dir = os.path.join(self.test_output_dir, "test_images")
        os.makedirs(test_data_dir, exist_ok=True)
        
        # Generate various test images
        self.test_images = self._create_test_images(test_data_dir)
        
        # Initialize detector
        try:
            self.detector = create_detector()
            if self.detector.is_ready():
                print("Detector initialized and ready")
                return True
            else:
                print("Detector created but models not loaded")
                print("Set model paths in config.py to run full tests")
                return False
        except Exception as e:
            print(f"Detector initialization failed: {e}")
            return False
    
    def _create_test_images(self, output_dir):
        """Create various test images for different scenarios"""
        test_images = []
        
        # Test Image 1: Clean product (should be GOOD)
        clean_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2.rectangle(clean_img, (150, 100), (490, 380), (180, 180, 220), -1)
        cv2.putText(clean_img, "GOOD PRODUCT", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        clean_path = os.path.join(output_dir, "good_product.jpg")
        cv2.imwrite(clean_path, clean_img)
        test_images.append(("good_product", clean_path, "GOOD"))
        
        # Test Image 2: Product with scratch (should be DEFECT)
        scratch_img = clean_img.copy()
        cv2.line(scratch_img, (200, 150), (400, 200), (50, 50, 50), 5)  # Dark scratch
        cv2.line(scratch_img, (250, 300), (350, 320), (40, 40, 40), 3)  # Another scratch
        scratch_path = os.path.join(output_dir, "scratched_product.jpg")
        cv2.imwrite(scratch_path, scratch_img)
        test_images.append(("scratched_product", scratch_path, "DEFECT"))
        
        # Test Image 3: Product with stain (should be DEFECT)
        stain_img = clean_img.copy()
        cv2.circle(stain_img, (300, 200), 30, (80, 60, 40), -1)  # Brown stain
        cv2.ellipse(stain_img, (400, 300), (25, 15), 45, 0, 360, (70, 50, 30), -1)  # Another stain
        stain_path = os.path.join(output_dir, "stained_product.jpg")
        cv2.imwrite(stain_path, stain_img)
        test_images.append(("stained_product", stain_path, "DEFECT"))
        
        # Test Image 4: Product with missing component (should be DEFECT)
        missing_img = clean_img.copy()
        cv2.rectangle(missing_img, (350, 180), (420, 220), (0, 0, 0), -1)  # Black hole
        cv2.putText(missing_img, "MISSING", (330, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        missing_path = os.path.join(output_dir, "missing_component.jpg")
        cv2.imwrite(missing_path, missing_img)
        test_images.append(("missing_component", missing_path, "DEFECT"))
        
        # Test Image 5: Damaged product (should be DEFECT)
        damaged_img = clean_img.copy()
        # Create irregular damage pattern
        pts = np.array([[250, 150], [280, 140], [320, 170], [300, 200], [260, 190]], np.int32)
        cv2.fillPoly(damaged_img, [pts], (20, 20, 20))
        damaged_path = os.path.join(output_dir, "damaged_product.jpg")
        cv2.imwrite(damaged_path, damaged_img)
        test_images.append(("damaged_product", damaged_path, "DEFECT"))
        
        print(f"Created {len(test_images)} test images")
        return test_images
    
    def test_single_image_processing(self):
        """Test 1: Single image processing functionality"""
        print("\nTest 1: Single Image Processing")
        print("-" * 50)
        
        if not self.detector or not self.detector.is_ready():
            print("Skipping - detector not ready")
            self.test_results['single_image'] = {'status': 'skipped', 'reason': 'detector not ready'}
            return False
        
        results = []
        processing_times = []
        
        for test_name, image_path, expected in self.test_images:
            print(f"Testing {test_name}...")
            
            start_time = time.time()
            try:
                result = self.detector.process_image(image_path, self.test_output_dir)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                if result:
                    decision = result['final_decision']
                    anomaly_score = result['anomaly_detection']['anomaly_score']
                    defects = result.get('detected_defect_types', [])
                    
                    # Check if result matches expectation
                    match = (decision == expected)
                    status = "PASS" if match else "UNEXPECTED"
                    
                    print(f"   {status} - Decision: {decision} (expected: {expected})")
                    print(f"   Anomaly Score: {anomaly_score:.4f}")
                    print(f"   Processing Time: {processing_time:.2f}s")
                    
                    if defects:
                        print(f"   Detected Defects: {', '.join(defects)}")
                    
                    results.append({
                        'test_name': test_name,
                        'expected': expected,
                        'actual': decision,
                        'match': match,
                        'anomaly_score': anomaly_score,
                        'processing_time': processing_time,
                        'defects': defects,
                        'has_visualization': result.get('visualization_path') is not None,
                        'has_report': result.get('report_path') is not None
                    })
                else:
                    print(f"   FAILED - No result returned")
                    results.append({
                        'test_name': test_name,
                        'expected': expected,
                        'actual': 'ERROR',
                        'match': False,
                        'error': 'No result returned'
                    })
                
            except Exception as e:
                print(f"   ERROR - {str(e)}")
                results.append({
                    'test_name': test_name,
                    'expected': expected,
                    'actual': 'ERROR',
                    'match': False,
                    'error': str(e)
                })
        
        # Calculate statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.get('match', False))
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        print(f"\nSingle Image Test Summary:")
        print(f"   Passed: {passed_tests}/{total_tests}")
        print(f"   Average Processing Time: {avg_processing_time:.2f}s")
        
        self.test_results['single_image'] = {
            'status': 'completed',
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests * 100,
            'avg_processing_time': avg_processing_time,
            'results': results
        }
        
        return passed_tests > 0
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\nGenerating Test Report...")
        
        total_time = time.time() - self.start_time
        
        # Create detailed report
        report = f"""
UNIFIED DEFECT DETECTION BACKEND - TEST REPORT
============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Test Duration: {total_time:.2f} seconds

SYSTEM INFORMATION
=================
Detector Ready: {self.detector.is_ready() if self.detector else False}
Device: {self.detector.device if self.detector else 'N/A'}
Test Images Created: {len(self.test_images)}
Output Directory: {self.test_output_dir}

TEST RESULTS SUMMARY
===================
"""
        
        # Add results for each test
        for test_name, result in self.test_results.items():
            status = result.get('status', 'unknown')
            report += f"\n{test_name.upper().replace('_', ' ')}:\n"
            report += f"  Status: {status}\n"
            
            if status == 'completed':
                if test_name == 'single_image':
                    report += f"  Success Rate: {result['success_rate']:.1f}%\n"
                    report += f"  Average Processing Time: {result['avg_processing_time']:.2f}s\n"
            elif status == 'skipped':
                report += f"  Reason: {result.get('reason', 'unknown')}\n"
        
        # Overall assessment
        completed_tests = sum(1 for r in self.test_results.values() if r.get('status') == 'completed')
        total_tests = len(self.test_results)
        
        report += f"""
OVERALL ASSESSMENT
=================
Completed Tests: {completed_tests}/{total_tests}
"""
        
        if completed_tests == total_tests:
            report += "Status: ALL TESTS PASSED - Backend is fully functional!\n"
        elif completed_tests >= total_tests - 1:
            report += "Status: MOSTLY FUNCTIONAL - Minor issues detected\n"
        else:
            report += "Status: ISSUES DETECTED - Check failed tests\n"
        
        report += f"""
RECOMMENDATIONS
==============
"""
        
        if not self.detector or not self.detector.is_ready():
            report += "1. Set correct model paths in config.py\n"
            report += "2. Ensure model files exist in models/ directory\n"
            report += "3. Install all required dependencies\n"
        else:
            single_image_result = self.test_results.get('single_image', {})
            if single_image_result.get('success_rate', 0) < 80:
                report += "1. Fine-tune detection thresholds\n"
                report += "2. Check model quality and training data\n"
        
        report += f"\nDetailed results saved in: {self.test_output_dir}\n"
        report += "Test completed successfully!\n"
        
        # Save report
        report_path = os.path.join(self.test_output_dir, "test_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save JSON results for further analysis
        json_path = os.path.join(self.test_output_dir, "test_results.json")
        with open(json_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"Test report saved: {report_path}")
        print(f"JSON results saved: {json_path}")
        
        return report_path
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        print("STARTING COMPREHENSIVE BACKEND TESTING")
        print("=" * 60)
        
        # Setup
        setup_success = self.setup_test_environment()
        
        # Run tests
        tests = [
            ("Single Image Processing", self.test_single_image_processing),
        ]
        
        test_passed = 0
        for test_name, test_func in tests:
            try:
                print(f"\n{'='*60}")
                success = test_func()
                if success:
                    test_passed += 1
            except Exception as e:
                print(f"{test_name} test crashed: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = {
                    'status': 'crashed',
                    'error': str(e)
                }
        
        # Generate final report
        print(f"\n{'='*60}")
        print("GENERATING FINAL REPORT")
        report_path = self.generate_test_report()
        
        # Final summary
        print(f"\nTESTING COMPLETED!")
        print(f"Tests passed: {test_passed}/{len(tests)}")
        print(f"Results directory: {self.test_output_dir}")
        print(f"Full report: {report_path}")
        
        return self.test_results


def main():
    """Main test runner"""
    tester = BackendTester()
    results = tester.run_all_tests()
    return results


if __name__ == "__main__":
    main()