import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --------------------- Step 1: Load and Prepare Dataset ---------------------
# Load CSV containing a column 'state' with values 0 (idle), 1 (busy), 2 (hybrid)
data = pd.read_csv("spectrum_states.csv")  # Ensure this CSV has one column named 'state'
states = data['state'].values

# Define sliding window size
window_size = 3
X, y = [], []

for i in range(window_size, len(states)):
    X.append(states[i - window_size:i])
    y.append(states[i])

X = np.array(X)
y = to_categorical(y, num_classes=3)  # One-hot encode target labels
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM: (samples, timesteps, features)

# --------------------- Step 2: Train-Test Split ---------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------- Step 3: Build LSTM Model ---------------------
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 output classes: Idle, Busy, Hybrid

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --------------------- Step 4: Train Model ---------------------
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# --------------------- Step 5: Evaluate Model ---------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --------------------- Step 6: Confusion Matrix ---------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=['Idle', 'Busy', 'Hybrid'], yticklabels=['Idle', 'Busy', 'Hybrid'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# --------------------- Step 7: Predict New State ---------------------
latest_sequence = np.array([1, 2, 0])  # Example input: last 3 known states
latest_sequence = latest_sequence.reshape((1, window_size, 1))

predicted = model.predict(latest_sequence)
predicted_class = np.argmax(predicted)

state_map = {0: "Idle", 1: "Busy", 2: "Hybrid"}
print("Predicted Next State:", state_map[predicted_class])
pip install <package-name> --trusted-host pypi.org --trusted-host files.pythonhosted.org --cert false