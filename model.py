import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, models
import numpy as np
from keras.callbacks import EarlyStopping


X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")

seq_length = 500 

model = models.Sequential([
    layers.Conv1D(32, kernel_size=5, input_shape=(seq_length, 4), padding = 'same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=5, padding = 'same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  
])


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

early_stopping = EarlyStopping(
    min_delta=0.001, 
    patience=5, 
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=64,
    callbacks=[early_stopping]
)

history_df = pd.DataFrame(history.history)
history_df.loc[5:, ['loss', 'val_loss']].plot()
plt.savefig("loss_plot.png")

history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.savefig("accuracy_plot.png")

print(f"Best Validation Loss: {history_df['val_loss'].min():.4f}")
print(f"Best Validation Accuracy: {history_df['val_binary_accuracy'].max():.4f}")

'''First training round
Best Validation Loss: 0.6742
Best Validation Accuracy: 0.5714

Second training round
Best Validation Loss: 0.6752
Best Validation Accuracy: 0.5772

Third training round
Best Validation Loss: 0.6931
Best Validation Accuracy: 0.5014

Fourth training round
Best Validation Loss: 0.6773
Best Validation Accuracy: 0.5663'''