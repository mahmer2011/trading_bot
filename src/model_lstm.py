import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_lstm_model(input_shape, dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    input_shape = (lookback, num_features)
    """
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units=32))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(units=16, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))  # binary classification

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


import os

def train_model(
    X_train, y_train, X_val, y_val,
    input_shape, model_save_path="models/lstm_model.h5",
    epochs=50, batch_size=64
):
    model = build_lstm_model(input_shape)
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ModelCheckpoint(model_save_path, monitor="val_accuracy", save_best_only=True)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return model, history
