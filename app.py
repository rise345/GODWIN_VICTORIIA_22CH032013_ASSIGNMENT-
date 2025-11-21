import os
import string
from typing import Dict, Tuple, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from google import genai
from google.genai.errors import APIError

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False

# Configuration
class Config:
    """Application configuration"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = "gemini-2.5-flash"
    PORT = int(os.environ.get("PORT", 5000))
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"


class GeminiClient:
    """Wrapper for Google Gemini API client"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client: Optional[genai.Client] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the Gemini client"""
        try:
            self.client = genai.Client(api_key=self.api_key)
            app.logger.info(f"✓ Gemini API initialized (Model: {self.model})")
        except Exception as e:
            app.logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from the LLM"""
        if not self.client:
            raise RuntimeError("Gemini client not initialized")
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        if response and response.text:
            return response.text
        else:
            return "I couldn't generate a response. Please try again."


# Initialize Gemini client
gemini_client: Optional[GeminiClient] = None

if Config.GEMINI_API_KEY:
    try:
        gemini_client = GeminiClient(Config.GEMINI_API_KEY, Config.MODEL_NAME)
    except Exception as e:
        app.logger.error(f"Error initializing Gemini: {e}")
else:
    app.logger.warning("GEMINI_API_KEY not found in environment variables!")

# --- Flask Routes ---


@app.route("/")
def index():
    """Serve the main HTML page"""
    return render_template("index.html")


class TextPreprocessor:
    """Text preprocessing utilities for NLP tasks"""
    
    @staticmethod
    def preprocess(text: str) -> Dict[str, any]:
        """
        Apply NLP preprocessing pipeline:
        1. Lowercasing
        2. Punctuation removal
        3. Tokenization
        
        Args:
            text: Raw input text
            
        Returns:
            Dictionary containing preprocessing steps and results
        """
        # Step 1: Store original
        original = text
        
        # Step 2: Convert to lowercase
        lowercased = text.lower()
        
        # Step 3: Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        no_punctuation = lowercased.translate(translator)
        
        # Step 4: Tokenization (whitespace splitting)
        tokens = no_punctuation.split()
        
        # Step 5: Reconstruct processed text
        processed = ' '.join(tokens)
        
        return {
            "original": original,
            "lowercased": lowercased,
            "punctuation_removed": no_punctuation,
            "tokens": tokens,
            "processed": processed
        }


@app.route("/api/ask", methods=["POST"])
def ask_question():
    """
    API endpoint for question answering with NLP preprocessing.
    
    Request JSON:
        {"question": str}
    
    Response JSON:
        {
            "question": str,
            "answer": str,
            "preprocessing": dict,
            "status": str
        }
    """
    # Validate Gemini client
    if not gemini_client:
        return jsonify({
            "error": "AI service not configured. Please check API key.",
            "status": "error"
        }), 503

    try:
        # Parse and validate request
        data = request.get_json()
        
        if not data or "question" not in data:
            return jsonify({
                "error": "Missing 'question' field in request body",
                "status": "error"
            }), 400

        question = data["question"].strip()

        if not question:
            return jsonify({
                "error": "Question cannot be empty",
                "status": "error"
            }), 400

        # Apply text preprocessing
        preprocessing = TextPreprocessor.preprocess(question)
        
        app.logger.info(f"Processing question: {question[:50]}...")

        # Generate AI response
        answer = gemini_client.generate_response(question)
        
        app.logger.info("Response generated successfully")

        return jsonify({
            "question": question,
            "answer": answer,
            "preprocessing": preprocessing,
            "status": "success"
        }), 200

    except APIError as e:
        app.logger.error(f"Gemini API Error: {str(e)}")
        return jsonify({
            "error": f"AI service error: {str(e)}",
            "status": "error"
        }), 500

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "An unexpected error occurred. Please try again.",
            "status": "error"
        }), 500


def main():
    """Run the Flask application"""
    print("\n" + "=" * 70)
    print("  NLP Question & Answering System")
    print("  Using Google Gemini LLM with Text Preprocessing")
    print("  Student: Osho Ayomide (23CG034142)")
    print("=" * 70)

    if not Config.GEMINI_API_KEY:
        print("\n⚠️  ERROR: GEMINI_API_KEY not found!")
        print("\nPlease create a .env file with your Google Gemini API key:")
        print("  GEMINI_API_KEY=your_api_key_here")
        print("\nGet your key from: https://makersuite.google.com/app/apikey")
        print("=" * 70 + "\n")
        return

    print(f"\n✓ Gemini Model: {Config.MODEL_NAME}")
    print(f"✓ Server: http://localhost:{Config.PORT}")
    print(f"✓ Debug Mode: {Config.DEBUG}")
    print("\nPress Ctrl+C to stop the server\n")

    app.run(
        debug=Config.DEBUG,
        host="0.0.0.0",
        port=Config.PORT
    )


if __name__ == "__main__":
    main()
