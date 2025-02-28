# 💬 Live Lip-Reading using Flask 

Real-time lip reading powered by deep learning! This Flask-based application processes live video input, extracts lip movements, and predicts spoken words using an advanced hybrid model (3D CNN + LSTM).

---

## 🚀 Features
- **Real-time lip reading**: Perform live lip reading using a webcam.
- **Face & Lip Extraction**: Uses Haar Cascade for face detection and dlib for lip extraction.
- **Deep Learning Model**: Hybrid 3D CNN + LSTM for sequence prediction.
- **Flask Web Interface**: Provides an easy-to-use interface for real-time predictions.

## 📌 Trained Model
The model used for real-time detection in this application was trained in [Project-X-Lip-Reading](https://github.com/sourishphate/Project-X-Lip). This repository contains the training pipeline and dataset preparation used to develop the deep learning model.

## 🔧 Running the Application
1. Clone the repository:
   ```bash
   git clone https://github.com/meekhumor/Lip-Reading.git
   cd Lip-Reading
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Open your browser and go to `http://127.0.0.1:5000/` to access the web interface.

## 🎯 How to Use
- Start the Flask application and grant access to the webcam.
- The system will detect lips in real-time and predict spoken words.
- Predicted words will be displayed on the web interface.

## 📂 File Structure
```
├── app.py                # Flask application
├── templates/            # HTML templates
├── models/               # Trained model files
├── utils/                # Helper functions for lip extraction & preprocessing
├── requirements.txt      # Required Python libraries
└── README.md             # Project documentation
```

## 🌟 Future Improvements
- **Multilingual Support**: Extend support for multiple languages.
- **Sentence-Level Predictions**: Improve accuracy by predicting full sentences.
- **Better Model Performance**: Enhance model accuracy and reduce latency.


## 👥 Contributors

- [Om Mukherjee](https://github.com/meekhumor)
- [Sourish Phate](https://github.com/sourishphate)



i
