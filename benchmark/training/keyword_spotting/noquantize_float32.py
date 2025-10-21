import tensorflow as tf
import os
import numpy as np
import argparse

import get_dataset as kws_data
import kws_util

if __name__ == '__main__':
  Flags, unparsed = kws_util.parse_command()

  print(f"Converting trained model {Flags.saved_model_path} to TFL model at {Flags.tfl_file_name}")
  model = tf.keras.models.load_model(Flags.saved_model_path)
  
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  
  fp32_tfl_file_name = Flags.tfl_file_name[:Flags.tfl_file_name.rfind('.')] + '_float32.tflite'
  tflite_float_model = converter.convert()
  with open(fp32_tfl_file_name, "wb") as fpo:
    num_bytes_written = fpo.write(tflite_float_model)
  print(f"Wrote {num_bytes_written} / {len(tflite_float_model)} bytes to tflite file")
