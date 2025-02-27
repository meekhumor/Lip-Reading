import numpy as np
import cv2
import dlib
import imutils
from imutils import face_utils
from skimage.transform import resize

def face_extractor(img):
    """Extract face from an image using Haar Cascade."""
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_classifier = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(image, 1.3, 5)

    # If faces are found, extract the first face
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cropped_image = image[y:y+h, x:x+w]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
            return cropped_image
    else:
        return None
    
def lips_extractor(img):
    """Extract lips region from a facial image using dlib landmarks."""
    predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

    image = imutils.resize(img, width=56)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    bbox = dlib.rectangle(0, 0, gray.shape[1], gray.shape[0])
    face_landmarks = predictor(gray, bbox)
    face_landmarks = face_utils.shape_to_np(face_landmarks)

    lip_image = None
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        if name == 'mouth':
            for (x, y) in face_landmarks[i:j]:
                (x, y, w, h) = cv2.boundingRect(np.array([face_landmarks[i:j]]))
                lip_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
                lip_image = imutils.resize(lip_image, width=500, inter=cv2.INTER_CUBIC)
                lip_image = cv2.cvtColor(lip_image, cv2.COLOR_BGR2GRAY)

    return lip_image

def preprocess_sequence(sequence, target_frames=28):
    """Preprocess a sequence of lip images for model prediction."""
    processed_frames = []
    
    for frame in sequence:
        frame = resize(frame, (100, 100))
        frame = 255 * frame
        frame = frame.astype(np.uint8)
        processed_frames.append(frame)
    
    # Pad to target length
    pad_array = [np.zeros((100, 100))]
    processed_frames.extend(pad_array * (target_frames - len(processed_frames)))
    
    # Convert to numpy array
    processed_frames = np.array(processed_frames)
    
    # Normalize
    np.seterr(divide='ignore', invalid='ignore')
    v_min = processed_frames.min(axis=(1, 2), keepdims=True)
    v_max = processed_frames.max(axis=(1, 2), keepdims=True)
    processed_frames = (processed_frames - v_min) / (v_max - v_min)
    processed_frames = np.nan_to_num(processed_frames)
    
    # Reshape for model
    processed_frames = processed_frames.reshape(1, target_frames, 100, 100, 1)
    
    return processed_frames