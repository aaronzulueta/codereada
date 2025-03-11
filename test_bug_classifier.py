"""
Test script to verify that the bug classifier works correctly with user-provided images.
"""
import sys
import os
from PIL import Image
import tensorflow as tf
import numpy as np

# Import functions from bugClassifier.py
try:
    from bugClassifier import extract_code_from_image, analyze_code_for_bugs
    print("Bug classifier module loaded successfully.")
except ImportError:
    print("Error: Bug classifier module not available.")
    sys.exit(1)

def test_bug_classifier(image_path):
    """Test the bug classifier with a user-provided image."""
    print(f"Testing bug classifier with image: {image_path}")
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    try:
        # Load the image
        image = Image.open(image_path)
        print("Image loaded successfully.")
        
        # Test OCR extraction
        extracted_code = extract_code_from_image(image)
        print("\nExtracted Code:")
        print("=" * 50)
        print(extracted_code)
        print("=" * 50)
        
        if not extracted_code:
            print("Warning: No code was extracted from the image.")
            return False
        
        # Try to load the bug classifier model
        try:
            from tensorflow import keras
            from tensorflow.keras.preprocessing.text import Tokenizer
            
            model_path = "bug_classification_model.keras"
            if not os.path.exists(model_path):
                print(f"Error: Bug classifier model not found: {model_path}")
                return False
            
            model = keras.models.load_model(model_path)
            print(f"Bug classifier model loaded successfully: {model_path}")
            
            # Get the input shape from the model
            input_shape = model.input_shape
            print(f"Model input shape: {input_shape}")
            
            # Create a tokenizer for testing
            tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
            
            # Sample Java code to fit tokenizer
            java_code_samples = [
                "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, World!\"); } }",
                "int sum(int a, int b) { return a + b; }",
                "@Override public void run() { System.out.println(\"Running thread\"); }"
            ]
            tokenizer.fit_on_texts(java_code_samples)
            
            # Analyze the code for bugs
            result = analyze_code_for_bugs(image, model, tokenizer)
            
            print("\nBug Detection Result:")
            print("=" * 50)
            if result['bug_detected']:
                print("❌ Bug Detected")
            else:
                print("✅ No Bug Detected")
            print("=" * 50)
            
            return True
            
        except Exception as e:
            print(f"Error during bug detection: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_bug_classifier.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    success = test_bug_classifier(image_path)
    
    if success:
        print("\nTest completed successfully.")
    else:
        print("\nTest failed.")
        sys.exit(1) 