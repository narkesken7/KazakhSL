import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define the labels
actions = np.array(['qaida', 'qalai', 'qandai', 'qashan', 'qai', 'kim', 'qai_jaqqa', 'qansha', 'ne', 'ne_ushin'])


# Define the data directory
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

# Initialize empty lists to store data and labels
data = []
labels = []

# Loop through each label and load the data
for label in actions:
    label_path = DATA_DIR[label]

    # Loop through subdirectories in the label directory
    for subdir in os.listdir(label_path):
        subdir_path = os.path.join(label_path, subdir)

        # Check if the subdirectory is a directory
        if os.path.isdir(subdir_path):

            # Initialize a list to store keypoints for this video
            keypoints_sequence = []

            # Loop through files in the subdirectory
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)

                # Check if the file is an .npy file
                if file_name.endswith('.npy'):
                    # Load the .npy file
                    keypoints = np.load(file_path)

                    # Append the keypoints to the sequence for this video
                    keypoints_sequence.append(keypoints)

            # Append the keypoints sequence to the data list and label index to the labels list
            data.append(keypoints_sequence)
            labels.append(actions.tolist().index(label))

# Convert data and labels to numpy arrays
data = np.array(data, dtype=object)  # Use dtype=object to store variable-length sequences
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Calculate the maximum sequence length in your dataset
max_sequence_length = max(len(seq) for seq in X_train)

# Pad sequences to have a fixed length
X_train = [np.pad(seq, ((0, max_sequence_length - len(seq)), (0, 0)), 'constant', constant_values=0) for seq in X_train]
X_test = [np.pad(seq, ((0, max_sequence_length - len(seq)), (0, 0)), 'constant', constant_values=0) for seq in X_test]

X_train = np.array(X_train, dtype='float32')
X_test = np.array(X_test, dtype='float32')

# Define the number of classes
num_classes = len(actions)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define the LSTM model
model = models.Sequential()
model.add(layers.LSTM(64, input_shape=(max_sequence_length, 1662), return_sequences=True))
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(64, activation='relu')))
model.add(layers.Flatten())
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define a callback to save the best model
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                                mode='max')

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint])
