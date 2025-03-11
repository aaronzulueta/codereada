import pytesseract
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2  # Added for image preprocessing
import json
import string
import tokenize
from io import BytesIO
from tensorflow.keras.preprocessing.sequence import pad_sequences


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


# Character-level representation
char_vocab = {char: idx for idx, char in enumerate(string.ascii_letters + string.digits + string.punctuation + " ", start=1)}

def char_to_int(code):
    """Convert code to character-level representation"""
    return [char_vocab.get(char, 0) for char in code]

# Token-level representation
token_vocab = {}

def tokenize_code(code):
    """Convert code to token-level representation"""
    tokens = []
    try:
        code_bytes = BytesIO(code.encode('utf-8'))
        for token in tokenize.tokenize(code_bytes.readline):
            token_str = token.string
            if token_str not in token_vocab:
                token_vocab[token_str] = len(token_vocab) + 1
            tokens.append(token_vocab[token_str])
    except:
        tokens.append(0)
    return tokens


# Extract text from an image using Pytesseract with OpenCV preprocessing
def extract_code_from_image(image):
    """
    Extract code from an image using OCR with OpenCV preprocessing.
    
    Args:
        image: PIL Image object
        
    Returns:
        str: Extracted text from the image
    """
    try:
        # Convert PIL Image to OpenCV format
        open_cv_image = np.array(image) 
        # Convert RGB to BGR (if image has 3 channels)
        if len(open_cv_image.shape) == 3:
            open_cv_image = open_cv_image[:, :, ::-1].copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply OCR with specific config
        extracted_text = pytesseract.image_to_string(gray, config="--psm 6")
        return extracted_text.strip()
    except Exception as e:
        print(f"Error in extract_code_from_image: {e}")
        import traceback
        traceback.print_exc()
        return ""  # Return empty string on error


def preprocess_code(code):
    """Preprocess code for bug classification model"""
    char_sequence = pad_sequences([char_to_int(code)], maxlen=1024)
    token_sequence = pad_sequences([tokenize_code(code)], maxlen=361)
    return [char_sequence, token_sequence]


def analyze_code_for_bugs(image, model=None, tokenizer=None):
    """
    Analyze code in an image for bugs using the new model.
    
    Args:
        image: PIL Image object
        model: Trained bug detection model
        tokenizer: Not used in the new implementation
        
    Returns:
        dict: Dictionary containing extracted code and bug detection result
    """
    # Initialize result dictionary with default values
    result = {
        'extracted_text': '',
        'bug_detected': None,
        'error_type': None
    }
    
    try:
        # Extract code from image
        extracted_code = extract_code_from_image(image)
        result['extracted_text'] = extracted_code
        
        # If no model or no extracted code, return early
        if model is None or not extracted_code:
            return convert_numpy_types(result)
        
        # Preprocess the extracted code
        input_data = preprocess_code(extracted_code)
        
        # Predict if there's a bug
        prediction = model.predict(input_data)
        print(f"Prediction shape: {prediction.shape}")
        
        # Get the predicted class
        predicted_class = np.argmax(prediction[0])
        
        # For now, we'll consider any error type as a bug
        # In a more sophisticated implementation, we could map the predicted class
        # to specific error types using a label encoder
        result['bug_detected'] = predicted_class > 0  # Class 0 might be "No Error"
        result['error_type'] = int(predicted_class)
        
        print(f"Predicted class: {predicted_class}")
        if result['bug_detected']:
            print("Classification: ❌ Bug Detected")
        else:
            print("Classification: ✅ No Bugs Detected")
                
    except Exception as e:
        print(f"Error in analyze_code_for_bugs: {e}")
        import traceback
        traceback.print_exc()
        # Result already has default values
    
    # Convert any NumPy types to Python native types before returning
    return convert_numpy_types(result)


# For standalone testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python bugClassifier.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        # Load image
        image = Image.open(image_path)
        
        # Load model if available
        try:
            model_path = "bug_classification_model2.keras"
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded bug classifier model: {model_path}")
            
            # Analyze code
            result = analyze_code_for_bugs(image, model)
            
            print("Extracted Code:\n", result['extracted_text'])
            
            if result['bug_detected']:
                print(f"❌ Bug Detected (Error Type: {result['error_type']})")
            else:
                print("✅ No Bug Detected")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Running in limited mode (OCR only)")
            import traceback
            traceback.print_exc()
            
            # Extract code only
            extracted_code = extract_code_from_image(image)
            print("Extracted Code:\n", extracted_code)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
