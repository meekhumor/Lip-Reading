import os
import numpy as np
import cv2
import tensorflow as tf
import base64
from flask import Flask, render_template, request, jsonify
from utils.preprocessing import face_extractor, lips_extractor, preprocess_sequence
from io import BytesIO
from PIL import Image

# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Words that the model can predict
WORDS = ['NULL', 'Begin', 'Choose', 'Connection', 'Navigation', 
         'Next', 'Previous', 'Start', 'Stop', 'Hello', 'Web']

# Load the model
model = tf.keras.models.load_model('models/3D_CNN_LSTM_words_final.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    # Get frames from request
    frames_data = request.json.get('frames', [])
    frames = []
    
    for frame_base64 in frames_data:
        # Decode base64 image
        frame_data = base64.b64decode(frame_base64.split(',')[1])
        frame_np = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        frames.append(frame)
    
    if len(frames) == 0:
        return jsonify({'error': 'No frames received'})
    
    # Process frames
    min_frames = 8
    max_frames = 28
    total_frames = len(frames)
    
    # Calculate the number of frames to extract
    num_frames = min(max_frames, max(min_frames, total_frames // 10))
    
    # Calculate the interval between frames
    interval = total_frames // num_frames
    
    sequence = []
    sequence_base64 = []
    
    for i in range(num_frames):
        frame = frames[i * interval]
        face_frame = face_extractor(frame)
        if face_frame is not None:
            lip_frame = lips_extractor(face_frame)
            if lip_frame is not None:
                sequence.append(lip_frame)
                
                # Convert lip frame to base64 for display in browser
                _, buffer = cv2.imencode('.jpg', lip_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                sequence_base64.append(f"data:image/jpeg;base64,{img_base64}")
    
    if len(sequence) == 0:
        return jsonify({'error': 'No valid lip sequences detected'})
    
    # Preprocess sequence for prediction
    processed_sequence = preprocess_sequence(sequence)
    
    # Make prediction
    prediction = model.predict(processed_sequence)
    
    # Process results
    percentages = [round(p * 100, 2) for p in prediction[0]]
    predictions = {WORDS[i]: percentages[i] for i in range(len(WORDS))}
    
    max_index = np.argmax(prediction)
    
    return jsonify({
        'word': WORDS[max_index],
        'confidence': percentages[max_index],
        'predictions': sorted(predictions.items(), key=lambda x: x[1], reverse=True),
        'sequence_images': sequence_base64
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)