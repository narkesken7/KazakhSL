import os
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import confusion_matrix, accuracy_score



# Directory structure
DATA_DIR = {
    'qaida': '/Users/keshubai/Desktop/CS/Python/Jupyter Notebook/V2_forTry3/_gde_q',
    'qalai': '/Users/keshubai/Desktop/CS/Python/Jupyter Notebook/V2_forTry3/_kak_q',
    'qandai': '/Users/keshubai/Desktop/CS/Python/Jupyter Notebook/V2_forTry3/_kakoi_q',
    'qashan': '/Users/keshubai/Desktop/CS/Python/Jupyter Notebook/V2_forTry3/_kogda_q',
    'qai': '/Users/keshubai/Desktop/CS/Python/Jupyter Notebook/V2_forTry3/_kotoriy_q',
    'kim': '/Users/keshubai/Desktop/CS/Python/Jupyter Notebook/V2_forTry3/_kto_q',
    'qai_jaqqa': '/Users/keshubai/Desktop/CS/Python/Jupyter Notebook/V2_forTry3/_kuda_q',
    'qansha': '/Users/keshubai/Desktop/CS/Python/Jupyter Notebook/V2_forTry3/_skolko_q',
    'ne': '/Users/keshubai/Desktop/CS/Python/Jupyter Notebook/V2_forTry3/_what_q',
    'ne_ushin': '/Users/keshubai/Desktop/CS/Python/Jupyter Notebook/V2_forTry3/_zachem_q'
}

# Define label map and actions
actions = np.array(['qaida', 'qalai', 'qandai', 'qashan', 'qai', 'kim', 'qai_jaqqa', 'qansha', 'ne', 'ne_ushin'])
label_map = {label: num for num, label in enumerate(actions)}

# Function to load keypoints for all videos and frames for a given action
def load_keypoints_for_action(directory, action_label):
    action = actions[action_label]
    keypoints_list = []

    for video_num in range(1, 201):
        for frame_num in range(max_sequence_length):
            video_dir = os.path.join(directory, f'_{action}_{video_num}')
            keypoints_path = os.path.join(video_dir, f'frame_{frame_num}.npy')

            if os.path.exists(keypoints_path):
                keypoints = np.load(keypoints_path)
                keypoints_list.append(keypoints)

    return keypoints_list

# Initialize empty lists to store keypoints and labels
X, y = [], []

max_sequence_length = 30  # Define max_sequence_length here

max_sequence_lengths = {action_label: 0 for action_label in range(len(actions))}
num_features = 30

# Calculate max_sequence_length for each action
for action, directory in DATA_DIR.items():
    action_label = label_map[action]  # Get the label for the current action

    for video_num in range(1, 201):
        for frame_num in range(max_sequence_length):
            # Load keypoints for the current action
            keypoints_for_action = load_keypoints_for_action(directory, action_label)

            max_sequence_lengths[action_label] = max(
                max_sequence_lengths[action_label],
                len(keypoints_for_action)
            )

# Get the overall maximum sequence length across all actions
max_sequence_length = max(max_sequence_lengths.values())

# Iterate through each action in DATA_DIR to load keypoints
for action, directory in DATA_DIR.items():
    action_label = label_map[action]  # Get the label for the current action

    for video_num in range(1, 201):
        for frame_num in range(max_sequence_length):
            # Load keypoints for the current action
            keypoints_for_action = load_keypoints_for_action(directory, action_label)

            # Padding to ensure all sequences have the same length
            if len(keypoints_for_action) < max_sequence_length:
                keypoints_for_action.extend([0] * (max_sequence_length - len(keypoints_for_action)))

            X.append(keypoints_for_action)
            y.append(action_label)

# Convert X and y to numpy arrays
X = np.array(X)
y = to_categorical(y, num_classes=len(actions))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


log_dir = os.path.join('Logs10')
tb_callback = TensorBoard(log_dir=log_dir)

# Define and compile the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(6, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[133]:
tensorboard --logdir == Logs10


model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

# Save the trained model
model.save('action_recognition_model.h5')

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate confusion matrix and accuracy
confusion = confusion_matrix(y_true, y_pred_classes)
accuracy = accuracy_score(y_true, y_pred_classes)

model.summary()

res = model.predict(X_test)
