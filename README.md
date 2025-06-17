# Unified Defect Detection System

A comprehensive backend system that combines **Anomalib** (anomaly detection) and **HRNet** (defect classification) for automated product quality inspection.

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Setup Models
Place your trained models in the `models/` directory:
- `models/anomalib_model.pt` - Your Anomalib model
- `models/hrnet_model.pth` - Your HRNet model

### 3. Configure
Edit `config.py` with your model paths:
```python
ANOMALIB_MODEL_PATH = MODELS_DIR / "your_anomalib_model.pt"
HRNET_MODEL_PATH = MODELS_DIR / "your_hrnet_model.pth"
```

### 4. Test System
```bash
python run_tests.py
```

### 5. Start API (for Flutter)
```bash
python api_server.py
```

## 🤝 Contributing

We welcome contributions! Follow this workflow to get started:

### First Time Setup
```bash
# 1. Clone the repository
git clone https://github.com/FitraRH/unified_defect_detection.git
cd unified_defect_detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Checkout to development branch
git checkout dev
git pull origin dev

# 4. Create your feature branch
git checkout -b feature/your-feature-name
# Example: git checkout -b feature/add-new-defect-type
```

### Development Workflow
```bash
# 1. Make sure you're on your feature branch
git branch

# 2. Make your changes...
# Edit files, add features, fix bugs

# 3. Run tests to ensure everything works
python run_tests.py

# 4. Add and commit changes
git add .
git commit -m "Add: descriptive commit message"
# Examples:
# git commit -m "Add: new defect classification for electronics"
# git commit -m "Fix: bounding box calculation accuracy"
# git commit -m "Update: improved API response format"

# 5. Push to your feature branch
git push -u origin feature/your-feature-name

# 6. Create Pull Request on GitHub
# Go to: https://github.com/FitraRH/unified_defect_detection
# Click "Compare & pull request"
# Target: dev branch (NOT main)
# Add description of your changes
```

### Keeping Your Branch Updated
```bash
# Update your local dev branch
git checkout dev
git pull origin dev

# Merge updates into your feature branch
git checkout feature/your-feature-name
git merge dev

# Or rebase for cleaner history
git rebase dev
```

### Before Submitting Pull Request
```bash
# 1. Run all tests
python run_tests.py

# 2. Check code style (if you have flake8/black installed)
flake8 .
black .

# 3. Update documentation if needed
# Edit README.md, docstrings, comments

# 4. Test API endpoints
python api_server.py
# Test with Postman or curl
```

### Branch Strategy
- **`main`** - Production-ready code
- **`dev`** - Development integration branch
- **`feature/*`** - Individual feature development
- **`hotfix/*`** - Urgent production fixes
- **`release/*`** - Release preparation

### Commit Message Guidelines
Use descriptive commit messages:

- **Add**: New features or files
  - `Add: real-time camera detection feature`
- **Fix**: Bug fixes
  - `Fix: memory leak in batch processing`
- **Update**: Improvements to existing features
  - `Update: enhanced anomaly detection accuracy`
- **Remove**: Deleting code or files
  - `Remove: deprecated visualization functions`
- **Docs**: Documentation changes
  - `Docs: update API endpoint examples`

### Code Review Process
1. Submit Pull Request to `dev` branch
2. Add descriptive title and description
3. Request review from maintainers
4. Address feedback and make changes
5. Once approved, your code will be merged

### Local Testing Commands
```bash
# Test specific functionality
python -c "from main import quick_detect; print(quick_detect('test_image.jpg'))"

# Test API server
python api_server.py
# In another terminal:
curl -X GET http://localhost:5000/api/health

# Run comprehensive tests
python run_tests.py

# Test with sample images
python -c "
from main import create_detector
detector = create_detector()
result = detector.process_image('path/to/test/image.jpg')
print(result)
"
```

## 📁 Project Structure

```
unified_defect_detection/
├── api_server.py              # Flask API for Flutter
├── run_tests.py               # Testing suite
├── main.py                    # Core detector class
├── config.py                  # Configuration
├── models/                    # AI models
├── core/                      # Detection logic
├── processors/                # Processing workflows
├── utils/                     # Utilities
└── outputs/                   # Results (auto-generated)
```

## 🎯 Main Commands

| Command | Purpose |
|---------|---------|
| `python api_server.py` | Start Flask API for Flutter |
| `python run_tests.py` | Test all functionality |
| `python -c "from main import quick_detect; print(quick_detect('image.jpg'))"` | Quick test |

## 📱 Flutter Integration

The system provides a REST API for Flutter apps:

- **Endpoint**: `http://localhost:5000/api/detect-image`
- **Method**: POST
- **Input**: Base64 image or multipart file
- **Output**: JSON with detection results

## 🔧 Usage Examples

### Single Image Detection
```python
from main import create_detector

detector = create_detector()
result = detector.process_image("product_image.jpg")
print(f"Decision: {result['final_decision']}")
```

### Batch Processing
```python
detector = create_detector()
batch_results = detector.process_batch("images_folder/")
```

### Real-time Camera
```python
detector = create_detector()
detector.start_camera()  # Press 'q' to quit
```

## 🎛️ Configuration

Key settings in `config.py`:

- **Model Paths**: Update with your model file locations
- **Thresholds**: Adjust sensitivity
  - `ANOMALY_THRESHOLD = 0.7` (higher = less sensitive)
  - `DEFECT_CONFIDENCE_THRESHOLD = 0.85`
- **Device**: Set to 'cuda' or 'cpu'
- **Defect Classes**: Customize detection categories

## 📊 Supported Defect Types

- **damaged** - Physical damage to product
- **missing_component** - Missing parts or components  
- **open** - Unsealed or open areas
- **scratch** - Surface scratches
- **stained** - Stains or discoloration

## 🔄 API Endpoints

### Health Check
```
GET /api/health
```

### Single Image Detection
```
POST /api/detect-image
Content-Type: application/json
{
  "image_base64": "data:image/jpeg;base64,..."
}
```

### Batch Detection
```
POST /api/batch-detect
Content-Type: application/json
{
  "images": ["base64_image1", "base64_image2", ...]
}
```

### Update Thresholds
```
POST /api/update-thresholds
Content-Type: application/json
{
  "anomaly_threshold": 0.7,
  "defect_threshold": 0.85
}
```

## 📈 Response Format

```json
{
  "status": "success",
  "final_decision": "DEFECT",
  "processing_time": 1.23,
  "anomaly_detection": {
    "anomaly_score": 0.8567,
    "decision": "DEFECT",
    "threshold_used": 0.7
  },
  "detected_defects": ["scratch", "stain"],
  "defect_count": 2,
  "bounding_boxes": [
    {
      "defect_type": "scratch",
      "x": 100,
      "y": 50,
      "width": 200,
      "height": 30,
      "center_x": 200,
      "center_y": 65,
      "area": 6000
    }
  ]
}
```

## 🧪 Testing

The system includes comprehensive testing:

```bash
python run_tests.py
```

Tests include:
- ✅ Single image processing
- ✅ Bounding box accuracy
- ✅ Performance benchmarks
- ✅ Error handling
- ✅ Edge cases

## 🔧 Customization

### Edit Detection Logic
- **Bounding boxes**: `core/detection.py` - `_extract_bounding_boxes()`
- **Detection criteria**: `core/detection.py` - `_analyze_defect_predictions()`
- **Thresholds**: `config.py` - Global variables

### Edit Visualization
- **Charts & plots**: `utils/visualization.py`
- **Reports**: `utils/reports.py`

### Edit Model Architecture
- **HRNet model**: `models/hrnet_model.py`
- **Model loading**: `models/model_loader.py`

## 📦 Dependencies

Core dependencies:
- PyTorch >= 1.12.0
- Anomalib >= 0.6.0
- OpenCV >= 4.6.0
- Flask >= 2.2.0 (for API)
- Albumentations >= 1.3.0

## 🚀 Deployment

### Local Development
```bash
python api_server.py
# Server runs on http://localhost:5000
```

### Production Deployment
```bash
# Docker
docker build -t defect-detection .
docker run -p 5000:5000 defect-detection

# Or with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

## 📱 Flutter Integration Example

```dart
// Flutter HTTP request
final response = await http.post(
  Uri.parse('http://your-server:5000/api/detect-image'),
  headers: {'Content-Type': 'application/json'},
  body: json.encode({
    'image_base64': base64Image
  }),
);

final result = json.decode(response.body);
print('Decision: ${result['final_decision']}');
```

## 🎯 Performance

Typical performance on modern hardware:
- **Single image**: 1-3 seconds
- **Throughput**: 20-60 images/minute
- **Memory usage**: 2-4 GB (GPU)
- **Model size**: 50-500 MB

## 🔒 Security Notes

For production deployment:
- Add authentication to API endpoints
- Validate input image size and format
- Implement rate limiting
- Use HTTPS for secure transmission

## 🐛 Issue Reporting

If you encounter issues:

1. **Check existing issues**: https://github.com/FitraRH/unified_defect_detection/issues
2. **Run diagnostics**:
   ```bash
   python run_tests.py
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
   ```
3. **Create new issue** with:
   - System information (OS, Python version)
   - Error message and stack trace
   - Steps to reproduce
   - Expected vs actual behavior

## 🏗️ Development Setup

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install flake8 black pytest
```

### IDE Configuration
Recommended VSCode extensions:
- Python
- GitLens
- Python Docstring Generator
- autoDocstring

## 📄 License

This project is licensed under the Apache License.

## 🤝 Support

For issues and questions:
1. Check the test results: `python run_tests.py`
2. Verify model paths in `config.py`
3. Check system requirements and dependencies
4. Create an issue on GitHub with detailed information

## 🔄 Updates

To update the system:
1. Pull latest changes: `git pull origin dev`
2. Update dependencies: `pip install -r requirements.txt`
3. Run tests: `python run_tests.py`
4. Restart API server: `python api_server.py`

## 👥 Contributors

Thanks to all contributors who make this project better!

- [Fitra RH](https://github.com/FitraRH) - Project Maintainer

---

**Ready for production use with Flutter, web, or desktop applications!** 🚀

### Quick Links
- 🐛 [Report Issues](https://github.com/FitraRH/unified_defect_detection/issues)
- 📖 [Wiki & Documentation](https://github.com/FitraRH/unified_defect_detection/wiki)
- 🔄 [Pull Requests](https://github.com/FitraRH/unified_defect_detection/pulls)
