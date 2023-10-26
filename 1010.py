import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load your trained model (replace with the actual model path)
model = load_model('best_model.h5')

# Define actions and colors
actions = np.array(['qaida', 'qalai', 'qandai', 'qashan', 'qai', 'kim', 'qai_jaqqa', 'qansha', 'ne', 'ne_ushin'])  # Replace with your action labels
# Define actions and colors

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128)]


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Function to perform MediaPipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw styled landmarks
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

# Function to extract keypoints from results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Function to visualize probabilities
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Initialize variables for prediction
sequences = []
sentence = []
predictions = []
threshold = 0.5

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Set up MediaPipe models
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    # Read frame from the camera
    ret, frame = cap.read()

    # Make detections using MediaPipe
    image, results = mediapipe_detection(frame, holistic)

    # Draw landmarks on the frame
    draw_styled_landmarks(image, results)

    # Keypoint extraction
    keypoints = extract_keypoints(results)
    sequences.append(keypoints)
    if len(sequences) > 8:
        sequences = sequences[-8:]

    # Create an input sequence with shape (1, 200, 1662)
    input_sequence = np.zeros((1, 200, 1662))

    # Concatenate the last 8 frames to match the input shape
    for i, seq in enumerate(reversed(sequences)):
        input_sequence[0, i * 25:i * 25 + 25, :] = seq

    # Make a prediction
    res = model.predict(input_sequence)[0]
    predictions.append(np.argmax(res))

    # Update the sentence
    if len(predictions) >= 5:
        unique_predictions = np.unique(predictions[-5:])
        if len(unique_predictions) == 1 and res[unique_predictions[0]] > threshold:
            word = actions[unique_predictions[0]]
            if len(sentence) == 0 or word != sentence[-1]:
                sentence.append(word)

    # Display the recognized words and accuracy
    recognized_sentence = ' '.join(sentence)
    accuracy = res[np.argmax(res)]
    cv2.putText(image, f'Accuracy: {accuracy:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, recognized_sentence, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Visualize the probabilities
    image = prob_viz(res, actions, image, colors)

    # Show the frame
    cv2.imshow('Sign Language Recognition', image)

    # Reset recognized words if 'r' is pressed
    key = cv2.waitKey(10)
    if key == ord('r'):
        sentence = []
        predictions = []

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
