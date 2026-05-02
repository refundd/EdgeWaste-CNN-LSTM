import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

seq_len = 3
features = 64
inputs = layers.Input(shape=(seq_len, features))
x = layers.LSTM(32, unroll=True)(inputs)
outputs = layers.Dense(3, activation='softmax')(x)

model = Model(inputs, outputs)

def rep_data():
    for _ in range(10):
        yield [np.random.randn(1, seq_len, features).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("test_lstm.tflite", "wb") as f:
    f.write(tflite_model)

import subprocess
import os
os.system("python3 -m tflite_tools.analyzer test_lstm.tflite || echo 'Please inspect test_lstm.tflite'")
