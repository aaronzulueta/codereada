from flask import Flask, render_template, request, jsonify
import os
import pytesseract
from PIL import Image
import io
import base64
import cv2  # Added for image preprocessing
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.saving import register_keras_serializable

# Register the custom metric with the correct namespace
@register_keras_serializable(package='Custom')
def classification_accuracy(y_true, y_pred):
    """Custom accuracy metric: Compare predicted vs. actual readability classification."""
    y_true_class = K.cast(y_true >= 2.5, dtype="float32")  
    y_pred_class = K.cast(y_pred >= 2.5, dtype="float32")
    return K.mean(K.equal(y_true_class, y_pred_class))

# Try to import TensorFlow with updated import paths
try:
    # For TensorFlow 2.6+
    try:
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
    # For older TensorFlow versions
    except ImportError:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    tensorflow_available = True
except ImportError:
    print("TensorFlow not available. Running in limited mode.")
    tensorflow_available = False

import numpy as np

# Import bug classifier functions
try:
    from bugClassifier import extract_code_from_image, analyze_code_for_bugs, preprocess_code
    bug_classifier_available = True
    print("Bug classifier module loaded successfully.")
except ImportError:
    bug_classifier_available = False
    print("Bug classifier module not available. Bug detection will be disabled.")

# Set Tesseract executable path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Sample Java code dataset for readability analysis (should be replaced with real training data)
java_code_samples = [
    "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, World!\"); } }",
    "int sum(int a, int b) { return a + b; }",
    "@Override public void run() { System.out.println(\"Running thread\"); }"
]

# Helper function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

# Initialize tokenizers and models if TensorFlow is available
if tensorflow_available:
    # Initialize for readability analysis
    readability_tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    readability_tokenizer.fit_on_texts(java_code_samples)  # Fit on Java snippets
    
    # Initialize for bug detection
    bug_model = None
    
    try:
        # Check if readability model exists, if not create a simple placeholder model
        readability_model_path = "readability_model2.keras"
        if os.path.exists(readability_model_path):
            # Load model with custom objects dictionary including both namespaced and non-namespaced versions
            custom_objects = {
                'classification_accuracy': classification_accuracy,
                'Custom>classification_accuracy': classification_accuracy
            }
            readability_model = tf.keras.models.load_model(
                readability_model_path,
                custom_objects=custom_objects
            )
            print(f"Loaded existing readability model: {readability_model_path}")
            
            # Get the input shape from the model
            readability_input_shape = readability_model.input_shape
            print(f"Readability model input shape: {readability_input_shape}")
        else:
            print(f"Readability model not found: {readability_model_path}. Creating a simple placeholder model.")
            # Create a simple placeholder model with the expected input/output shape
            readability_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(200,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(5, activation='softmax')
            ])
            readability_model.compile(optimizer='adam', 
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy', classification_accuracy])
            readability_model.save(readability_model_path)
            readability_input_shape = (None, 200)
            
        # Check if bug classifier model exists
        if bug_classifier_available:
            bug_model_path = "bug_classification_model2.keras"  # Updated to use the new model
            if os.path.exists(bug_model_path):
                bug_model = tf.keras.models.load_model(bug_model_path)
                print(f"Loaded existing bug classifier model: {bug_model_path}")
            else:
                print(f"Bug classifier model not found: {bug_model_path}. Bug detection will be disabled.")
    except Exception as e:
        print(f"Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        tensorflow_available = False

def extract_text_from_image(image):
    """Extracts text from an image using OCR with OpenCV preprocessing."""
    try:
        if bug_classifier_available:
            # Use the function from bugClassifier.py
            return extract_code_from_image(image)
        else:
            # Fallback to local implementation with OpenCV preprocessing
            # Convert PIL Image to OpenCV format
            open_cv_image = np.array(image) 
            # Convert RGB to BGR (if image has 3 channels)
            if len(open_cv_image.shape) == 3:
                open_cv_image = open_cv_image[:, :, ::-1].copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply OCR with specific config
            text = pytesseract.image_to_string(gray, config="--psm 6")
            return text.strip()  # Remove unnecessary whitespace
    except Exception as e:
        print(f"Error extracting text: {e}")
        import traceback
        traceback.print_exc()
        return ""  # Return empty string on error

def preprocess_text_for_readability(text, max_length=200):
    """
    Convert Java code into a numerical tensor for readability model input.
    The readability model expects a shape of (None, 200).
    """
    if not tensorflow_available:
        return None
        
    if not text:  # If extracted text is empty
        return np.zeros((1, max_length))  # Return an empty input with the correct shape
    
    try:
        # Tokenize text
        sequences = readability_tokenizer.texts_to_sequences([text])  # Convert text to numerical sequences
        
        if not sequences or all(len(seq) == 0 for seq in sequences):  
            return np.zeros((1, max_length))  # Handle empty sequence case
        
        # Pad sequences to match the expected input shape
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
        
        # Ensure the shape is (1, max_length)
        return padded_sequences
    except Exception as e:
        print(f"Error in preprocess_text_for_readability: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((1, max_length))  # Return an empty input with the correct shape on error

@app.route('/')
def index():
    bug_detection_available = bug_classifier_available and tensorflow_available and bug_model is not None
    return render_template('index.html', 
                          tensorflow_available=tensorflow_available,
                          bug_detection_available=bug_detection_available)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Read the image
            image = Image.open(file.stream)
            
            # Initialize result dictionary
            result = {
                'extracted_text': '',
                'readability_score': None,
                'bug_detected': None,
                'error_type': None
            }
            
            # If bug classifier is available, use it for both text extraction and bug detection
            if bug_classifier_available and tensorflow_available and bug_model is not None:
                try:
                    # Use the analyze_code_for_bugs function from bugClassifier.py
                    bug_analysis = analyze_code_for_bugs(image, bug_model)
                    
                    # Check if bug_analysis contains the expected keys
                    if isinstance(bug_analysis, dict) and 'extracted_text' in bug_analysis:
                        result['extracted_text'] = bug_analysis.get('extracted_text', '')
                        result['bug_detected'] = bug_analysis.get('bug_detected')
                        result['error_type'] = bug_analysis.get('error_type')
                        print(f"Bug detection result: {result['bug_detected']}, Error type: {result['error_type']}")
                    else:
                        # Fallback to direct extraction if bug_analysis doesn't have the expected structure
                        result['extracted_text'] = extract_text_from_image(image)
                except Exception as e:
                    print(f"Error in bug analysis: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback to direct extraction
                    result['extracted_text'] = extract_text_from_image(image)
            else:
                # Extract text using the local function
                result['extracted_text'] = extract_text_from_image(image)
            
            # If TensorFlow is available, predict readability score
            if tensorflow_available and result['extracted_text']:
                try:
                    # Readability analysis
                    X_input_readability = preprocess_text_for_readability(result['extracted_text'])
                    
                    if X_input_readability is not None:
                        print(f"Input shape for readability model: {X_input_readability.shape}")
                        readability_prediction = readability_model.predict(X_input_readability)
                        print(f"Readability prediction shape: {readability_prediction.shape}, value: {readability_prediction}")
                        
                        # Get raw prediction value without rounding or classification
                        raw_score = float(readability_prediction[0][0] if len(readability_prediction.shape) > 1 else readability_prediction[0])
                        result['readability_score'] = raw_score
                        print(f"Raw readability score: {raw_score}")
                except Exception as e:
                    print(f"Error during readability prediction: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Convert NumPy types to Python native types for JSON serialization
            result = convert_numpy_types(result)
            
            return jsonify(result)
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 