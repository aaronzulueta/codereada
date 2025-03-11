# Code Analysis Tool

This application uses OCR to extract code from images and analyze both its readability and detect potential bugs.

## Setup Instructions

### 1. Install Dependencies

Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

The dependencies include:
- flask==2.3.3
- numpy==1.23.5 (critical - newer versions like NumPy 2.x cause compatibility issues)
- pillow==9.5.0
- pytesseract==0.3.10
- tensorflow==2.15.0 (optional)

If you encounter NumPy compatibility issues, run:
```
python fix_numpy.py
```

### 2. Install Tesseract OCR

For pytesseract to work, you need to install Tesseract OCR on your system:

#### Windows:
1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install it with default options
3. Add the Tesseract installation directory to your PATH environment variable (typically `C:\Program Files\Tesseract-OCR`)

#### Mac:
```
brew install tesseract
```

#### Linux:
```
sudo apt install tesseract-ocr
```

### 3. Model Files

The application requires two model files for full functionality:
- `readability_model.keras`: For analyzing code readability
- `bug_classification_model.keras`: For detecting bugs in code

Place these files in the root directory of the application. If either model is missing, the corresponding functionality will be disabled.

### 4. Running the Web Application

To run the web-based UI:

```bash
python app.py
```

Or use the run script:
```bash
python run.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

The web interface allows you to:
1. Upload an image containing code via drag-and-drop or file selection
2. View the extracted code with syntax highlighting
3. See the readability score (if TensorFlow and the readability model are available)
4. Check if bugs are detected in the code (if TensorFlow and the bug classifier model are available)
5. Copy the extracted code to clipboard

### 5. Application Modes

The application can run in three modes:
- **Full mode**: With TensorFlow and both models installed, the application can extract code, analyze readability, and detect bugs.
- **Partial mode**: With TensorFlow and only one model installed, the application will provide only the available analysis.
- **Limited mode**: Without TensorFlow, the application can only extract code from images.

## Features

- **OCR Code Extraction**: Extract code from images using Tesseract OCR
- **Readability Analysis**: Analyze code readability using a CNN model (requires TensorFlow)
- **Bug Detection**: Detect potential bugs in code using a CNN model (requires TensorFlow)
- **Responsive UI**: Modern, responsive web interface that works on desktop and mobile
- **Drag & Drop**: Easy file uploading with drag and drop support
- **Syntax Highlighting**: Code is displayed with syntax highlighting for better readability

## Troubleshooting

### Tesseract OCR Issues

If you encounter errors related to pytesseract not finding the Tesseract executable, you can specify the path in the app.py file:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as needed
```

### TensorFlow Issues

If you encounter issues with TensorFlow installation, you can:

1. Run the application in limited mode (it will automatically detect if TensorFlow is not available)
2. Try installing a different version of TensorFlow compatible with your Python version:
   ```
   pip install tensorflow==2.15.0  # For Python 3.11 on Windows
   ```

### Dependency Conflicts

If you encounter dependency conflicts, you can use the provided scripts:
- `fix_dependencies.py`: Fixes issues with numpy, pandas, and pytesseract
- `fix_tensorflow.py`: Attempts to install a compatible version of TensorFlow
- `fix_numpy.py`: Downgrades NumPy to a compatible version (1.23.5) 