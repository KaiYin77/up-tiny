#!/usr/bin/env python
"""
Script to convert WAV files to C array format for KWS model testing.
Based on the processing pipeline from wav2kws_tflite.py and make_bin_files.py.
"""

import tensorflow as tf
import numpy as np
import os
import sys
import argparse
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
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "spectrogram_length": spectrogram_length,
        "dct_coefficient_count": dct_coefficient_count,
        "fingerprint_size": fingerprint_size,
        "label_count": label_count,
        "sample_rate": flags.sample_rate,
    }

def load_audio_tf(wav_file, flags):
    """Load audio using librosa to support both 16-bit and 32-bit WAV files"""
    try:
        import librosa
        
        # Calculate desired samples from model settings
        model_settings = prepare_model_settings(len(WORD_LABELS), flags)
        desired_samples = model_settings["desired_samples"]
        
        # Use librosa to load the audio
        audio, sr = librosa.load(
            wav_file,
            sr=flags.sample_rate,
            mono=True,
            duration=flags.clip_duration_ms / 1000,
        )
        
        # Ensure we have the exact length needed
        if len(audio) < desired_samples:
            # Pad if too short
            audio = np.pad(audio, (0, desired_samples - len(audio)), "constant")
        elif len(audio) > desired_samples:
            # Trim if too long
            audio = audio[:desired_samples]
        
        # Normalize to [-1.0, 1.0] range
        if np.max(np.abs(audio)) > 0:  # Avoid division by zero
            audio = audio / np.max(np.abs(audio))
        
        # Convert numpy array to TensorFlow tensor
        wav = tf.convert_to_tensor(audio, dtype=tf.float32)
        
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
    wav = wav / (max_val + 1e-6)  # Scale to [0, 1], avoid division by zero
    
    # Apply time offset (matching the training pipeline)
    padded_wav = tf.pad(wav, [[2, 2]], mode="CONSTANT")
    shifted_wav = tf.slice(padded_wav, [2], [model_settings["desired_samples"]])
    
    # Compute STFT with Hann window
    stfts = tf.signal.stft(
        shifted_wav,
        frame_length=model_settings["window_size_samples"],
        frame_step=model_settings["window_stride_samples"],
        window_fn=tf.signal.hann_window,
    )
    
    spectrogram = tf.abs(stfts)
    
    # Compute Mel spectrogram
    num_spectrogram_bins = tf.shape(stfts)[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=model_settings["sample_rate"],
        lower_edge_hertz=20.0,
        upper_edge_hertz=4000.0,
    )
    
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate([40]))
    
    # Compute log-mel spectrogram and extract MFCCs
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., : model_settings["dct_coefficient_count"]]
    
    # Reshape to [spectrogram_length, dct_coefficient_count, 1]
    processed_features = tf.reshape(
        mfccs,
        [
            model_settings["spectrogram_length"],
            model_settings["dct_coefficient_count"],
            1,
        ],
    )
    
    # Add batch dimension
    features = tf.expand_dims(processed_features, axis=0)
    return features

def wav_to_c_array(wav_file, flags, output_file=None):
    """Convert WAV file to quantized MFCC features and output as C array"""
    
    # Load audio
    wav = load_audio_tf(wav_file, flags)
    if wav is None:
        print("Failed to load audio file")
        return None
    
    # Extract MFCC features
    features = extract_mfcc_features_tf(wav, flags)
    
    # Use quantization parameters from your model
    input_scale = 0.5847029089927673
    input_zero_point = 83
    
    # Quantize features to int8 with proper rounding
    features_np = features.numpy()
    # Apply quantization with proper rounding
    quantized_float = features_np / input_scale + input_zero_point
    features_q = np.round(quantized_float).astype(np.int8)
    
    # Debug: Show quantization process for first 5 values
    print(f"[QUANTIZATION] Original float (first 5): {features_np.flatten()[:5]}")
    print(f"[QUANTIZATION] After scale+offset: {quantized_float.flatten()[:5]}")
    print(f"[QUANTIZATION] After rounding: {features_q.flatten()[:5]}")
    print(f"[QUANTIZATION] As hex: {[hex(x & 0xFF) for x in features_q.flatten()[:5]]}")
    
    # Flatten the array
    flat_features = features_q.flatten()
    
    # Get the word from filename
    word = os.path.splitext(os.path.basename(wav_file))[0].lower()
    
    # Generate C array
    c_array_size = len(flat_features)
    
    # Create C array string in the requested format
    c_code = f"""#include "kws_mock_input_data.h"
// {word}
const int8_t g_kws_inputs[kNumKwsTestInputs][kKwsInputSize] = {{
"""
    
    # Add array elements as hex values (16 per line)
    c_code += "    "
    for i in range(0, len(flat_features), 16):
        line_elements = flat_features[i:i+16]
        line_str = "        " + ", ".join(f"0x{val & 0xFF:02X}" for val in line_elements)
        if i + 16 < len(flat_features):
            line_str += ","
        c_code += line_str + "\n"
    
    c_code += "    };\n"
    
    # Output to file or stdout
    if output_file:
        with open(output_file, 'w') as f:
            f.write(c_code)
        print(f"C array written to {output_file}")
    else:
        print(c_code)
    
    print(f"Array size: {c_array_size}")
    print(f"Feature shape: {features_q.shape}")
    print(f"Quantization scale: {input_scale}, zero_point: {input_zero_point}")
    
    return c_code

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert WAV files to C arrays for KWS testing")
    parser.add_argument("wav_file", help="Path to WAV file to convert")
    parser.add_argument("--output", "-o", help="Output C file (default: stdout)")
    
    args = parser.parse_args()
    
    # Get default flags from kws_util
    flags, _ = kws_util.parse_command()
    
    # Convert WAV to C array
    wav_to_c_array(args.wav_file, flags, args.output)

if __name__ == "__main__":
    main()