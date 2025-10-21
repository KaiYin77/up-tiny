#!/usr/bin/env python
"""
Inference script for quantized KWS model using INT8 TFLite model.
Combines patterns from eval_quantized_model.py, quantize.py, and infer.py.
"""

import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import librosa
import kws_util

# Labels used in the model
WORD_LABELS = [
    "Down", "Go", "Left", "No", "Off", "On", "Right",
    "Stop", "Up", "Yes", "Silence", "Unknown"
]

def prepare_model_settings(label_count, flags):
    """Calculates common settings needed for all models."""
    desired_samples = int(flags.sample_rate * flags.clip_duration_ms / 1000)
    
    # For MFCC features
    dct_coefficient_count = flags.dct_coefficient_count
    window_size_samples = int(flags.sample_rate * flags.window_size_ms / 1000)
    window_stride_samples = int(flags.sample_rate * flags.window_stride_ms / 1000)
    length_minus_window = desired_samples - window_size_samples
    
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
        fingerprint_size = flags.dct_coefficient_count * spectrogram_length
    
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': flags.sample_rate,
    }

def load_audio_tf(wav_file, flags):
    """Load audio using librosa to support both 16-bit and 32-bit WAV files"""
    try:
        # Calculate desired samples from model settings
        model_settings = prepare_model_settings(len(WORD_LABELS), flags)
        desired_samples = model_settings['desired_samples']
        
        # Use librosa to load the audio
        audio, sr = librosa.load(wav_file, sr=flags.sample_rate, mono=True, 
                                duration=flags.clip_duration_ms / 1000)
        
        # Ensure we have the exact length needed
        if len(audio) < desired_samples:
            # Pad if too short
            audio = np.pad(audio, (0, desired_samples - len(audio)), 'constant')
        elif len(audio) > desired_samples:
            # Trim if too long
            audio = audio[:desired_samples]
        
        # Normalize to [-1.0, 1.0] range
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Convert numpy array to TensorFlow tensor
        wav = tf.convert_to_tensor(audio, dtype=tf.float32)
        
        print(f"Successfully loaded audio from {wav_file}")
        return wav
        
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

def extract_mfcc_features_tf(wav, flags):
    """Extract MFCC features using TensorFlow to match the training pipeline exactly"""
    # Prepare model settings
    model_settings = prepare_model_settings(len(WORD_LABELS), flags)
    
    # Normalize audio
    wav = tf.cast(wav, tf.float32)
    max_val = tf.reduce_max(tf.abs(wav))
    wav = wav / (max_val + 1e-6)
    
    print(f"[PREPROCESSING] Raw samples (first 10): {wav.numpy()[:10]}")
    
    # Apply time offset (matching the training pipeline)
    padded_wav = tf.pad(wav, [[2, 2]], mode='CONSTANT')
    shifted_wav = tf.slice(padded_wav, [2], [model_settings['desired_samples']])
    
    # Compute STFT with Hann window
    stfts = tf.signal.stft(
        shifted_wav,
        frame_length=model_settings['window_size_samples'],
        frame_step=model_settings['window_stride_samples'],
        window_fn=tf.signal.hann_window
    )
    
    spectrogram = tf.abs(stfts)
    print(f"[PREPROCESSING] Spectrogram shape: {spectrogram.shape}, max: {np.max(spectrogram.numpy())}")
    
    # Compute Mel spectrogram
    num_spectrogram_bins = tf.shape(stfts)[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=model_settings['sample_rate'],
        lower_edge_hertz=20.0,
        upper_edge_hertz=4000.0
    )
    
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate([40]))
    
    # Compute log-mel spectrogram and extract MFCCs
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    print(f"[PREPROCESSING] Log-mel spectrogram max: {np.max(log_mel_spectrogram.numpy())}")
    
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :model_settings['dct_coefficient_count']]
    
    print(f"[PREPROCESSING] MFCCs shape: {mfccs.shape}, max: {np.max(mfccs.numpy())}")
    
    # Reshape to [spectrogram_length, dct_coefficient_count, 1]
    processed_features = tf.reshape(mfccs, [
        model_settings['spectrogram_length'],
        model_settings['dct_coefficient_count'],
        1
    ])
    
    # Add batch dimension
    features = tf.expand_dims(processed_features, axis=0)
    
    print(f"[PREPROCESSING] Final features shape: {features.shape}")
    return features

def infer_quantized_model(tflite_model_path, wav_file, flags):
    """Run inference with quantized TFLite model"""
    
    # Load audio and extract features
    wav = load_audio_tf(wav_file, flags)
    if wav is None:
        print("Failed to load audio file")
        return None
    
    features = extract_mfcc_features_tf(wav, flags)
    
    # Load TFLite model
    print(f"\nLoading quantized TFLite model from: {tflite_model_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Model loaded successfully!")
    print(f"Input details: {input_details[0]}")
    print(f"Output details: {output_details[0]}")
    
    # Get quantization parameters
    input_scale, input_zero_point = input_details[0]["quantization"]
    print(f"Input quantization - scale: {input_scale}, zero_point: {input_zero_point}")
    
    # Prepare input data
    features_np = features.numpy().astype(np.float32)
    
    # Print input tensor info
    print("\n=== Input Tensor ===")
    flat_input = features_np.flatten()
    first_5_input = flat_input[:5] if len(flat_input) >= 5 else flat_input
    print(f"Input Shape: {features_np.shape}")
    print(f"Input First 5 values (float32): {first_5_input}")
    print(f"Input Min: {np.min(features_np):.6f}, Max: {np.max(features_np):.6f}")
    
    # Handle quantization based on input type
    if input_details[0]['dtype'] == np.float32:
        print("Using float32 input")
        interpreter.set_tensor(input_details[0]['index'], features_np)
    elif input_details[0]['dtype'] == np.int8:
        print("Quantizing input to int8")
        # Quantize the input (matching eval_quantized_model.py)
        dat_q = np.array(features_np/input_scale + input_zero_point, dtype=np.int8)
        print(f"Quantized input - min: {np.min(dat_q)}, max: {np.max(dat_q)}")
        print(f"Quantized First 5 values (int8): {dat_q.flatten()[:5]}")
        interpreter.set_tensor(input_details[0]['index'], dat_q)
    else:
        raise ValueError(f"TFLite file has input dtype {input_details[0]['dtype']}. Only np.int8 and np.float32 are supported")
    
    print("\n=== Running Inference ===")
    # Run inference
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Raw output shape: {output_data.shape}")
    print(f"Raw output values: {output_data[0]}")
    
    # Get prediction
    prediction = output_data[0]
    predicted_class = np.argmax(prediction)
    
    # For quantized outputs, we might need to dequantize
    if output_details[0]['dtype'] == np.int8:
        output_scale, output_zero_point = output_details[0]["quantization"]
        print(f"Output quantization - scale: {output_scale}, zero_point: {output_zero_point}")
        # Dequantize output
        prediction_dequant = output_scale * (prediction.astype(np.float32) - output_zero_point)
        print(f"Dequantized output: {prediction_dequant}")
        confidence = prediction_dequant[predicted_class]
        all_scores = prediction_dequant
    else:
        confidence = prediction[predicted_class]
        all_scores = prediction
    
    return predicted_class, WORD_LABELS[predicted_class], confidence, all_scores

def main():
    # Get arguments from kws_util
    flags, _ = kws_util.parse_command()
    
    # Add our own arguments
    parser = argparse.ArgumentParser(description='KWS Quantized Inference')
    parser.add_argument('--wav_file', 
                       default='bcm-wavs/go.wav',
                       help='Path to WAV file to test')
    parser.add_argument('--tflite_model_path',
                       default='trained_models/kws_ref_model.tflite',
                       help='Path to quantized TFLite model file')
    
    # Parse our arguments
    test_args = parser.parse_args()
    
    print(f"Loading quantized TFLite model: {test_args.tflite_model_path}")
    print(f"Testing audio file: {test_args.wav_file}")
    
    # Run inference
    result = infer_quantized_model(test_args.tflite_model_path, test_args.wav_file, flags)
    
    if result is None:
        print("Inference failed")
        return
    
    # Display results
    class_id, label, confidence, all_scores = result
    
    print("\n=== Final Prediction Results ===")
    print(f"Predicted Class ID: {class_id}")
    print(f"Predicted Label: {label}")
    print(f"Confidence: {confidence:.6f}")
    
    print("\nAll Class Probabilities:")
    for i, score in enumerate(all_scores):
        print(f"  {WORD_LABELS[i]}: {score:.6f}")

if __name__ == '__main__':
    main()