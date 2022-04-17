import tensorflow as tf

def create_LSTM_model(X_train,mask,classes):

    # input
    input = tf.keras.Input(shape=X_train.shape[1:])
    mask = tf.keras.layers.Masking(mask_value=mask, input_shape=(X_train.shape[1:]))(input)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh'))(mask)
    x = tf.keras.layers.Dropout(0.5)(x)  
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, activation='tanh'))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation='selu')(x)
    # output
    output = tf.keras.layers.Dense(classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=[input], outputs=[output])
    return model