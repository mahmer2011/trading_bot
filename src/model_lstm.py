# model_lstm.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_3class_lstm_model(input_shape):
    """
    Build and compile a 3-class LSTM model.
    input_shape should be a tuple: (lookback, num_features).
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # First LSTM block
    model.add(LSTM(128, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Second LSTM block
    model.add(LSTM(64))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Dense layers
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Final 3-class output
    model.add(Dense(3, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
