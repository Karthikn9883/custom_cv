import cv2
import numpy as np
import os
import sys
import yaml
import time
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from projection_utils import ProjectionSystem
from camera_visualizer import CameraVisualizer
from performance_optimizer import PerformanceOptimizer

class SystemTestValidator:
    """
    Comprehensive test and validation suite for the 2D-3D mapping system.
    Tests RTSP connectivity, calibration accuracy, projection precision, and performance.
    """
    
    def __init__(self, rtsp_url: str = None, config_dir: str = None):
        if config_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(script_dir, '..', 'configs')
        
        self.config_dir = config_dir
        self.rtsp_url = rtsp_url
        self.test_results = {}
        self.performance_metrics = {}
        
        # Load RTSP URL from config if not provided
        if self.rtsp_url is None:
            self.load_rtsp_config()
        
        print(f"\\n{'='*70}")
        print("2D-3D MAPPING SYSTEM - COMPREHENSIVE TEST SUITE")
        print(f"{'='*70}")
        print(f"Config directory: {self.config_dir}")
        print(f"RTSP URL: {self.rtsp_url}")
        print(f"{'='*70}\\n")
    
    def load_rtsp_config(self):
        """Load RTSP URL from config."""
        config_path = os.path.join(self.config_dir, 'main_config.yaml')
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.rtsp_url = config.get('input_source', '0')
        except Exception as e:
            print(f"Warning: Could not load RTSP config: {e}")
            self.rtsp_url = 'rtsp://192.168.1.234:554/avstream/channel=1/stream=0.sdp'
    
    def test_system_requirements(self) -> Dict[str, bool]:
        """Test system requirements and dependencies."""
        print("Testing system requirements...")
        results = {}
        
        try:
            # Test Python version
            import sys
            python_version = sys.version_info
            results['python_version'] = python_version >= (3, 7)
            print(f"  Python version: {python_version.major}.{python_version.minor}.{python_version.micro} {'✓' if results['python_version'] else '✗'}")
            
            # Test OpenCV
            import cv2
            opencv_version = cv2.__version__
            results['opencv'] = True
            print(f"  OpenCV version: {opencv_version} ✓")
            
            # Test CUDA support
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0
            results['cuda_support'] = cuda_devices > 0
            print(f"  CUDA support: {'✓' if results['cuda_support'] else '✗'} ({cuda_devices} devices)")
            
            # Test NumPy
            import numpy as np
            numpy_version = np.__version__
            results['numpy'] = True
            print(f"  NumPy version: {numpy_version} ✓")
            
            # Test matplotlib
            import matplotlib
            matplotlib_version = matplotlib.__version__
            results['matplotlib'] = True
            print(f"  Matplotlib version: {matplotlib_version} ✓")
            
            # Test trimesh
            import trimesh
            results['trimesh'] = True
            print(f"  Trimesh available ✓")
            
            # Test PyYAML
            import yaml
            results['pyyaml'] = True
            print(f"  PyYAML available ✓")
            
        except Exception as e:
            print(f"  Error testing requirements: {e}")
            results['error'] = str(e)
        
        self.test_results['system_requirements'] = results
        return results
    
    def test_rtsp_connectivity(self) -> Dict[str, any]:
        """Test RTSP stream connectivity and quality."""
        print(f"\\nTesting RTSP connectivity: {self.rtsp_url}")
        results = {}
        
        try:
            # Initialize performance optimizer for better RTSP handling
            optimizer = PerformanceOptimizer()
            
            # Test connection
            start_time = time.time()
            if self.rtsp_url.isdigit():
                cap = cv2.VideoCapture(int(self.rtsp_url))
            else:
                cap = optimizer.create_optimized_video_capture(self.rtsp_url)
            
            connection_time = time.time() - start_time
            results['connection_time_seconds'] = connection_time
            results['connected'] = cap.isOpened()
            
            if not results['connected']:
                print(f"  Connection: ✗ Failed to connect")
                results['error'] = "Failed to connect to RTSP stream"
                return results
            
            print(f"  Connection: ✓ Connected in {connection_time:.2f}s")
            
            # Get stream properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            results['resolution'] = (width, height)
            results['fps'] = fps
            
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps:.1f}")
            
            # Test frame capture
            frame_times = []
            successful_frames = 0
            total_attempts = 10
            
            print(f"  Testing frame capture ({total_attempts} frames)...")
            
            for i in range(total_attempts):
                start = time.time()
                ret, frame = cap.read()
                frame_time = time.time() - start
                
                if ret:
                    successful_frames += 1
                    frame_times.append(frame_time)
                
                time.sleep(0.1)
            
            results['frame_capture_success_rate'] = successful_frames / total_attempts
            results['avg_frame_time_ms'] = np.mean(frame_times) * 1000 if frame_times else 0
            results['max_frame_time_ms'] = np.max(frame_times) * 1000 if frame_times else 0
            
            print(f"  Frame capture success rate: {results['frame_capture_success_rate']:.1%}")
            print(f"  Average frame time: {results['avg_frame_time_ms']:.1f}ms")
            print(f"  Max frame time: {results['max_frame_time_ms']:.1f}ms")
            
            # Quality assessment
            if successful_frames > 5:
                ret, sample_frame = cap.read()
                if ret:
                    # Check for frame quality issues
                    gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    results['frame_sharpness'] = laplacian_var
                    
                    # Check brightness
                    mean_brightness = np.mean(gray)
                    results['mean_brightness'] = mean_brightness
                    
                    print(f"  Frame sharpness: {laplacian_var:.1f}")
                    print(f"  Mean brightness: {mean_brightness:.1f}")
            
            cap.release()
            results['overall_quality'] = 'excellent' if results['frame_capture_success_rate'] > 0.95 else 'good' if results['frame_capture_success_rate'] > 0.8 else 'poor'
            
        except Exception as e:
            print(f"  Error testing RTSP: {e}")
            results['error'] = str(e)
            results['connected'] = False
        
        self.test_results['rtsp_connectivity'] = results
        return results
    
    def test_calibration_files(self) -> Dict[str, bool]:
        """Test presence and validity of calibration files."""
        print(f"\\nTesting calibration files...")
        results = {}
        
        # List of required calibration files
        required_files = {
            'camera_intrinsics.yaml': 'Camera intrinsic parameters',
            'camera_extrinsics.yaml': 'Camera extrinsic parameters', 
            'floor_plane.yaml': 'Floor plane definition',
            'main_config.yaml': 'Main configuration'
        }
        
        for filename, description in required_files.items():
            filepath = os.path.join(self.config_dir, filename)
            exists = os.path.exists(filepath)
            results[filename] = exists
            
            if exists:
                try:
                    with open(filepath, 'r') as f:
                        data = yaml.safe_load(f)
                    print(f"  {description}: ✓ Valid YAML")
                    
                    # Specific validation for each file
                    if filename == 'camera_intrinsics.yaml':
                        has_matrix = 'camera_matrix' in data and 'distortion_coefficients' in data
                        results[f'{filename}_valid'] = has_matrix
                        print(f"    Camera matrix present: {'✓' if has_matrix else '✗'}")
                    
                    elif filename == 'camera_extrinsics.yaml':
                        has_extrinsics = 'rotation_vector' in data and 'translation_vector' in data
                        results[f'{filename}_valid'] = has_extrinsics
                        print(f"    Rotation/translation present: {'✓' if has_extrinsics else '✗'}")
                    
                    elif filename == 'floor_plane.yaml':
                        has_plane = 'floor_plane' in data
                        results[f'{filename}_valid'] = has_plane
                        print(f"    Floor plane data present: {'✓' if has_plane else '✗'}")
                        
                except Exception as e:
                    print(f"  {description}: ✗ Invalid YAML - {e}")
                    results[f'{filename}_valid'] = False
            else:
                print(f"  {description}: ✗ Missing")
        
        self.test_results['calibration_files'] = results
        return results
    
    def test_projection_system(self) -> Dict[str, any]:
        """Test the 2D-3D projection system."""
        print(f"\\nTesting projection system...")
        results = {}
        
        try:
            # Initialize projection system
            proj_system = ProjectionSystem(self.config_dir)
            results['initialized'] = True
            print(f"  Projection system initialization: ✓")
            
            # Check calibration status
            is_calibrated = proj_system.is_calibrated()
            results['fully_calibrated'] = is_calibrated
            print(f"  Full calibration status: {'✓' if is_calibrated else '✗'}")
            
            if is_calibrated:
                # Test projection accuracy with known points
                test_points_2d = [
                    (320, 240),  # Center
                    (100, 100),  # Top-left region
                    (540, 100),  # Top-right region
                    (320, 380),  # Bottom center
                ]
                
                projection_errors = []
                successful_projections = 0
                
                print(f"  Testing projection accuracy with {len(test_points_2d)} test points...")
                
                for i, pixel_2d in enumerate(test_points_2d):
                    # Project 2D to 3D
                    world_3d = proj_system.project_2d_to_3d(pixel_2d)
                    
                    if world_3d is not None:
                        # Project back to 2D
                        pixel_2d_back = proj_system.project_3d_to_2d(world_3d)
                        
                        if pixel_2d_back is not None:
                            # Calculate reprojection error
                            error = np.sqrt((pixel_2d[0] - pixel_2d_back[0])**2 + 
                                          (pixel_2d[1] - pixel_2d_back[1])**2)
                            projection_errors.append(error)
                            successful_projections += 1
                            
                            print(f"    Point {i+1}: {pixel_2d} → 3D → {pixel_2d_back} (error: {error:.2f}px)")
                
                results['successful_projections'] = successful_projections
                results['projection_success_rate'] = successful_projections / len(test_points_2d)
                
                if projection_errors:
                    results['mean_reprojection_error'] = np.mean(projection_errors)
                    results['max_reprojection_error'] = np.max(projection_errors)
                    
                    print(f"  Mean reprojection error: {results['mean_reprojection_error']:.2f}px")
                    print(f"  Max reprojection error: {results['max_reprojection_error']:.2f}px")
                    
                    # Quality assessment
                    if results['mean_reprojection_error'] < 2.0:
                        projection_quality = 'excellent'
                    elif results['mean_reprojection_error'] < 5.0:
                        projection_quality = 'good'
                    else:
                        projection_quality = 'poor'
                    
                    results['projection_quality'] = projection_quality
                    print(f"  Projection quality: {projection_quality}")
            
            else:
                print(f"  ⚠️  System not fully calibrated - skipping projection tests")
                results['projection_quality'] = 'uncalibrated'
            
        except Exception as e:
            print(f"  Error testing projection system: {e}")
            results['error'] = str(e)
            results['initialized'] = False
        
        self.test_results['projection_system'] = results
        return results
    
    def test_camera_visualization(self) -> Dict[str, bool]:
        """Test camera visualization components."""
        print(f"\\nTesting camera visualization...")
        results = {}
        
        try:
            # Test camera visualizer initialization
            camera_viz = CameraVisualizer(self.config_dir)
            results['camera_visualizer_init'] = True
            print(f"  Camera visualizer initialization: ✓")
            
            # Test camera position calculation
            camera_pos = camera_viz.get_camera_position()
            results['camera_position_available'] = camera_pos is not None
            print(f"  Camera position calculation: {'✓' if camera_pos is not None else '✗'}")
            
            if camera_pos is not None:
                print(f"    Position: ({camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f})")
            
            # Test camera direction calculation
            camera_dir = camera_viz.get_camera_direction()
            results['camera_direction_available'] = camera_dir is not None
            print(f"  Camera direction calculation: {'✓' if camera_dir is not None else '✗'}")
            
            # Test FOV calculation
            h_fov, v_fov = camera_viz.get_field_of_view()
            results['fov_calculation'] = True
            print(f"  Field of view calculation: ✓ ({h_fov:.1f}° × {v_fov:.1f}°)")
            
            # Test frustum generation
            frustum_corners = camera_viz.generate_frustum_corners()
            results['frustum_generation'] = frustum_corners is not None
            print(f"  Frustum generation: {'✓' if frustum_corners is not None else '✗'}")
            
        except Exception as e:
            print(f"  Error testing camera visualization: {e}")
            results['error'] = str(e)
            results['camera_visualizer_init'] = False
        
        self.test_results['camera_visualization'] = results
        return results
    
    def test_performance_optimization(self) -> Dict[str, any]:
        """Test performance optimization features."""
        print(f"\\nTesting performance optimizations...")
        results = {}
        
        try:
            # Initialize performance optimizer
            optimizer = PerformanceOptimizer()
            results['optimizer_init'] = True
            print(f"  Performance optimizer initialization: ✓")
            
            # Test system detection
            system_info = optimizer.system_info
            results['system_detection'] = len(system_info) > 0
            print(f"  System detection: ✓")
            print(f"    RAM: {system_info.get('total_ram_gb', 'Unknown')}GB")
            print(f"    CPU cores: {system_info.get('cpu_cores', 'Unknown')}")
            
            # Test CUDA detection
            cuda_available = optimizer.cuda_available
            results['cuda_detection'] = True
            print(f"  CUDA detection: ✓ ({'Available' if cuda_available else 'Not available'})")
            
            # Test optimizations
            optimizations = optimizer.apply_all_optimizations()
            results['optimizations_applied'] = len(optimizations)
            print(f"  Optimizations applied: {len(optimizations)}")
            
            # Test RTSP optimization settings
            rtsp_settings = optimizer.optimize_rtsp_settings()
            results['rtsp_optimization'] = len(rtsp_settings) > 0
            print(f"  RTSP optimization settings: ✓")
            
        except Exception as e:
            print(f"  Error testing performance optimization: {e}")
            results['error'] = str(e)
            results['optimizer_init'] = False
        
        self.test_results['performance_optimization'] = results
        return results
    
    def run_comprehensive_test(self) -> Dict[str, any]:
        """Run all tests and return comprehensive results."""
        print(f"Starting comprehensive system test...\\n")
        start_time = time.time()
        
        # Run all tests
        test_functions = [
            self.test_system_requirements,
            self.test_calibration_files,
            self.test_projection_system,
            self.test_camera_visualization,
            self.test_performance_optimization,
            self.test_rtsp_connectivity,  # Run this last as it takes time
        ]
        
        for test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                print(f"Error in {test_func.__name__}: {e}")
        
        # Generate overall assessment
        total_time = time.time() - start_time
        self.test_results['test_duration_seconds'] = total_time
        self.test_results['timestamp'] = datetime.now().isoformat()
        
        # Calculate overall system health
        system_health = self.calculate_system_health()
        self.test_results['overall_health'] = system_health
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    def calculate_system_health(self) -> Dict[str, any]:
        """Calculate overall system health score."""
        health = {
            'score': 0,
            'max_score': 0,
            'status': 'unknown',
            'critical_issues': [],
            'recommendations': []
        }
        
        # System requirements (20 points)
        req_results = self.test_results.get('system_requirements', {})
        if req_results.get('python_version'): health['score'] += 5
        if req_results.get('opencv'): health['score'] += 5
        if req_results.get('numpy'): health['score'] += 3
        if req_results.get('matplotlib'): health['score'] += 3
        if req_results.get('trimesh'): health['score'] += 2
        if req_results.get('pyyaml'): health['score'] += 2
        health['max_score'] += 20
        
        # CUDA support bonus
        if req_results.get('cuda_support'):
            health['score'] += 5
            health['max_score'] += 5
        else:
            health['recommendations'].append("Install CUDA support for better performance")
        
        # Calibration files (25 points)
        cal_results = self.test_results.get('calibration_files', {})
        required_files = ['camera_intrinsics.yaml', 'camera_extrinsics.yaml', 'floor_plane.yaml', 'main_config.yaml']
        for file in required_files:
            if cal_results.get(file):
                health['score'] += 5
                if cal_results.get(f'{file}_valid', True):
                    health['score'] += 1.25
                else:
                    health['critical_issues'].append(f"Invalid {file}")
            else:
                health['critical_issues'].append(f"Missing {file}")
        health['max_score'] += 25
        
        # Projection system (25 points)
        proj_results = self.test_results.get('projection_system', {})
        if proj_results.get('initialized'):
            health['score'] += 10
            if proj_results.get('fully_calibrated'):
                health['score'] += 15
                quality = proj_results.get('projection_quality', 'poor')
                if quality == 'excellent':
                    health['score'] += 10
                elif quality == 'good':
                    health['score'] += 5
            else:
                health['critical_issues'].append("System not fully calibrated")
        else:
            health['critical_issues'].append("Projection system failed to initialize")
        health['max_score'] += 35
        
        # RTSP connectivity (15 points)
        rtsp_results = self.test_results.get('rtsp_connectivity', {})
        if rtsp_results.get('connected'):
            health['score'] += 10
            success_rate = rtsp_results.get('frame_capture_success_rate', 0)
            health['score'] += success_rate * 5
        else:
            health['critical_issues'].append("RTSP connection failed")
        health['max_score'] += 15
        
        # Calculate percentage and status
        health['percentage'] = (health['score'] / health['max_score']) * 100 if health['max_score'] > 0 else 0
        
        if health['percentage'] >= 90:
            health['status'] = 'excellent'
        elif health['percentage'] >= 75:
            health['status'] = 'good'
        elif health['percentage'] >= 50:
            health['status'] = 'fair'
        else:
            health['status'] = 'poor'
        
        return health
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print(f"\\n{'='*70}")
        print("COMPREHENSIVE TEST RESULTS SUMMARY")
        print(f"{'='*70}")
        
        health = self.test_results.get('overall_health', {})
        status = health.get('status', 'unknown').upper()
        score = health.get('score', 0)
        max_score = health.get('max_score', 100)
        percentage = health.get('percentage', 0)
        
        print(f"\\nOVERALL SYSTEM HEALTH: {status}")
        print(f"Score: {score:.1f}/{max_score} ({percentage:.1f}%)")
        
        # Critical issues
        critical_issues = health.get('critical_issues', [])
        if critical_issues:
            print(f"\\nCRITICAL ISSUES ({len(critical_issues)}):")
            for issue in critical_issues:
                print(f"  ❌ {issue}")
        else:
            print(f"\\n✅ No critical issues found!")
        
        # Recommendations
        recommendations = health.get('recommendations', [])
        if recommendations:
            print(f"\\nRECOMMENDATIONS ({len(recommendations)}):")
            for rec in recommendations:
                print(f"  💡 {rec}")
        
        # Detailed results
        print(f"\\nDETAILED TEST RESULTS:")
        
        test_categories = {
            'system_requirements': 'System Requirements',
            'calibration_files': 'Calibration Files',
            'projection_system': 'Projection System',
            'camera_visualization': 'Camera Visualization',
            'performance_optimization': 'Performance Optimization',
            'rtsp_connectivity': 'RTSP Connectivity'
        }
        
        for key, name in test_categories.items():
            results = self.test_results.get(key, {})
            if results:
                success_indicators = [k for k, v in results.items() if isinstance(v, bool) and v]
                total_tests = len([k for k, v in results.items() if isinstance(v, bool)])
                if total_tests > 0:
                    success_rate = len(success_indicators) / total_tests
                    status_icon = "✅" if success_rate >= 0.8 else "⚠️" if success_rate >= 0.5 else "❌"
                    print(f"  {status_icon} {name}: {len(success_indicators)}/{total_tests} tests passed")
        
        # Performance metrics
        duration = self.test_results.get('test_duration_seconds', 0)
        print(f"\\nTest completed in {duration:.1f} seconds")
        print(f"Timestamp: {self.test_results.get('timestamp', 'Unknown')}")
        
        print(f"\\n{'='*70}\\n")
    
    def save_test_report(self, filename: str = None):
        """Save detailed test report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_test_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"✅ Test report saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving test report: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive System Test and Validator")
    parser.add_argument('--source', type=str, help='RTSP URL to test')
    parser.add_argument('--config-dir', type=str, help='Configuration directory')
    parser.add_argument('--save-report', action='store_true', help='Save detailed test report to file')
    args = parser.parse_args()
    
    # Run comprehensive test
    validator = SystemTestValidator(rtsp_url=args.source, config_dir=args.config_dir)
    results = validator.run_comprehensive_test()
    
    # Save report if requested
    if args.save_report:
        validator.save_test_report()
    
    # Return appropriate exit code
    health = results.get('overall_health', {})
    percentage = health.get('percentage', 0)
    
    if percentage >= 75:
        print("✅ System ready for production use!")
        exit(0)
    elif percentage >= 50:
        print("⚠️  System functional but needs attention")
        exit(1)
    else:
        print("❌ System has critical issues - calibration/setup required")
        exit(2)

if __name__ == "__main__":
    main()