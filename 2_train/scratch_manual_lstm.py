import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def build_manual_lstm(seq_len=3, features=64, units=32):
    inputs = layers.Input(shape=(seq_len, features))
    
    x_0 = inputs[:, 0, :]
    x_1 = inputs[:, 1, :]
    x_2 = inputs[:, 2, :]
    
    zero_dense_h = layers.Dense(units, kernel_initializer='zeros', bias_initializer='zeros', trainable=False, name="init_h")
    zero_dense_c = layers.Dense(units, kernel_initializer='zeros', bias_initializer='zeros', trainable=False, name="init_c")
    
    h_state = zero_dense_h(x_0)
    c_state = zero_dense_c(x_0)
    
    def lstm_cell(x_t, h_prev, c_prev, step):
        concat = layers.Concatenate(name=f"concat_{step}")([x_t, h_prev])
        f_t = layers.Dense(units, activation='sigmoid', name=f"f_{step}")(concat)
        i_t = layers.Dense(units, activation='sigmoid', name=f"i_{step}")(concat)
        o_t = layers.Dense(units, activation='sigmoid', name=f"o_{step}")(concat)
        c_prime = layers.Dense(units, activation='tanh', name=f"cprime_{step}")(concat)
        
        f_mul_c = layers.Multiply(name=f"f_mul_c_{step}")([f_t, c_prev])
        i_mul_cprime = layers.Multiply(name=f"i_mul_cprime_{step}")([i_t, c_prime])
        c_t = layers.Add(name=f"c_t_{step}")([f_mul_c, i_mul_cprime])
        
        tanh_c_t = layers.Activation('tanh', name=f"tanh_c_t_{step}")(c_t)
        h_t = layers.Multiply(name=f"h_t_{step}")([o_t, tanh_c_t])
        return h_t, c_t

    h_state, c_state = lstm_cell(x_0, h_state, c_state, 0)
    h_state, c_state = lstm_cell(x_1, h_state, c_state, 1)
    h_state, c_state = lstm_cell(x_2, h_state, c_state, 2)
    
    outputs = layers.Dense(3, activation='softmax')(h_state)
    return Model(inputs, outputs)

model = build_manual_lstm()

def rep_data():
    for _ in range(10):
        yield [np.random.randn(1, 3, 64).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("test_manual_lstm.tflite", "wb") as f:
    f.write(tflite_model)
print("SUCCESSFUL CONVERSION")
