#!/usr/bin/env python
"""
Script to test WAV files with KWS model using TensorFlow MFCC features
matching the training pipeline preprocessing exactly.
Supports both 16-bit and 32-bit WAV files.
"""

import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import glob
import kws_util
import librosa

# Labels used in the model
WORD_LABELS = [
    "Down",
    "Go",
    "Left",
    "No",
    "Off",
    "On",
    "Right",
    "Stop",
    "Up",
    "Yes",
    "Silence",
    "Unknown",
]


def check_if_quantized(model_path):
    """Check if a TFLite model is quantized"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check if input is quantized
    input_quantized = input_details[0]["dtype"] == np.int8

    # Check if output is quantized
    output_quantized = output_details[0]["dtype"] == np.int8

    print(f"Model: {model_path}")
    print(f"Input tensor type: {input_details[0]['dtype']}")
    print(f"Input quantized: {input_quantized}")
    print(f"Output tensor type: {output_details[0]['dtype']}")
    print(f"Output quantized: {output_quantized}")

    return input_quantized or output_quantized


def prepare_model_settings(label_count, flags):
    """Calculates common settings needed for all models.
    Args:
        label_count: How many classes are to be recognized.
        flags: Namespace containing model parameters.
    Returns:
        Dictionary containing common settings.
    """
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
        # Calculate desired samples from model settings
        model_settings = prepare_model_settings(len(WORD_LABELS), flags)
        desired_samples = model_settings["desired_samples"]

        # Use librosa to load the audio (supports both 16-bit and 32-bit WAV)
        import librosa

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

        print(
            f"Successfully loaded audio from {wav_file} with librosa (supports 32-bit WAV)"
        )
        return wav

    except Exception as e:
        print(f"Error loading audio: {e}")
        import traceback

        traceback.print_exc()
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


def test_wav_file(model, wav_file, flags):
    """Test a single WAV file and return prediction using TensorFlow preprocessing"""
    # Load audio
    wav = load_audio_tf(wav_file, flags)

    if wav is None:
        print("Failed to load audio file")
        return None

    # Extract features
    features = extract_mfcc_features_tf(wav, flags)

    # Make prediction
    if hasattr(flags, "is_tflite") and flags.is_tflite:
        # If using TFLite model
        interpreter = tf.lite.Interpreter(model_path=flags.tfl_file_name)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Convert to numpy for TFLite
        features_np = features.numpy()

        # Prepare input data (quantize if needed)
        if input_details[0]["dtype"] == np.int8:
            input_scale, input_zero_point = input_details[0]["quantization"]
            features_q = np.array(
                features_np / input_scale + input_zero_point, dtype=np.int8
            )
            interpreter.set_tensor(input_details[0]["index"], features_q)
        else:
            interpreter.set_tensor(input_details[0]["index"], features_np)

        # Run inference
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]["index"])

        if output_details[0]["dtype"] == np.int8:
            # Dequantize output if needed
            output_scale, output_zero_point = output_details[0]["quantization"]
            output = (output.astype(np.float32) - output_zero_point) * output_scale
    else:
        # If using full Keras model
        output = model.predict(features, verbose=0)

    # Get prediction
    prediction = output[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    return predicted_class, WORD_LABELS[predicted_class], confidence, prediction


def get_ground_truth_from_filename(wav_file):
    """Extract ground truth label from filename

    Handles various filename formats:
    - simple.wav (e.g., "down.wav")
    - complex format: 0cb74144_nohash_1-stop-7.wav

    For complex format, the ground truth is after the hyphen and before the last hyphen or period
    """
    filename = os.path.basename(wav_file).lower()
    # Remove extension
    filename = os.path.splitext(filename)[0]

    # Case 1: Simple format where filename is the label
    for label in WORD_LABELS:
        if label.lower() == filename:
            return label

    # Case 2: Complex format with hyphens (e.g., 0cb74144_nohash_1-stop-7)
    if "-" in filename:
        # Split by hyphens
        parts = filename.split("-")

        # Try to extract label part (usually after first hyphen)
        if len(parts) >= 2:
            # The label should be after the first hyphen
            potential_label = parts[1]

            # If there's a number after the label (e.g., stop-7), remove it
            if "-" in potential_label:
                potential_label = potential_label.split("-")[0]

            # Check if the extracted part matches a known label
            for label in WORD_LABELS:
                if label.lower() == potential_label:
                    return label

    # Case 3: If no patterns match, check if any label is contained in the filename
    for label in WORD_LABELS:
        if label.lower() in filename:
            return label

    # If no match found
    print(f"Warning: Could not extract ground truth from {filename}")
    return "Unknown"


def test_all_wavs(model, wav_dir, flags):
    """Test all WAV files in the directory and compare with ground truth"""
    # Get all WAV files
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))

    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return

    print(f"Found {len(wav_files)} WAV files")

    # Results storage
    results = []
    correct_count = 0
    class_confusion = {label: {"total": 0, "correct": 0} for label in WORD_LABELS}
    confusion_matrix = np.zeros((len(WORD_LABELS), len(WORD_LABELS)), dtype=int)

    # Process each file
    for wav_file in wav_files:
        ground_truth = get_ground_truth_from_filename(wav_file)

        # Test the file
        result = test_wav_file(model, wav_file, flags)

        if result is None:
            print(f"Failed to process {wav_file}")
            continue

        class_id, predicted_label, confidence, all_scores = result

        # Check if prediction matches ground truth
        is_correct = predicted_label.lower() == ground_truth.lower()
        if is_correct:
            correct_count += 1

        # Update confusion matrix
        if ground_truth in WORD_LABELS:
            ground_truth_idx = WORD_LABELS.index(ground_truth)
            predicted_idx = WORD_LABELS.index(predicted_label)
            confusion_matrix[ground_truth_idx][predicted_idx] += 1

            # Update per-class statistics
            if ground_truth in class_confusion:
                class_confusion[ground_truth]["total"] += 1
                if is_correct:
                    class_confusion[ground_truth]["correct"] += 1

        # Save results
        results.append(
            {
                "filename": os.path.basename(wav_file),
                "ground_truth": ground_truth,
                "predicted": predicted_label,
                "confidence": confidence,
                "correct": is_correct,
            }
        )

        # Print individual result
        print(f"File: {os.path.basename(wav_file)}")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Predicted: {predicted_label} (confidence: {confidence:.4f})")
        print(f"  Result: {'✓' if is_correct else '✗'}")
        print()

    # Print summary
    accuracy = correct_count / len(wav_files) if wav_files else 0
    print("\nSummary:")
    print(f"Total files: {len(wav_files)}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")

    # Print per-class accuracy
    print("\nPer-Class Accuracy:")
    for label in WORD_LABELS:
        if label in class_confusion and class_confusion[label]["total"] > 0:
            class_acc = (
                class_confusion[label]["correct"] / class_confusion[label]["total"]
            )
            print(
                f"  {label}: {class_acc:.2%} ({class_confusion[label]['correct']}/{class_confusion[label]['total']})"
            )

    # Print confusion matrix
    print("\nConfusion Matrix:")
    # Print header
    print("  " + " ".join(f"{label[:4]:<5}" for label in WORD_LABELS))
    # Print rows
    for i, label in enumerate(WORD_LABELS):
        row = confusion_matrix[i]
        print(f"{label[:4]:<4} " + " ".join(f"{count:<5}" for count in row))

    return results


def main():
    # First, get the arguments from kws_util
    flags, _ = kws_util.parse_command()

    # Now add our own additional arguments for testing
    parser = argparse.ArgumentParser(description="Test WAV files with KWS model")
    parser.add_argument(
        "--wav_file", default="bcm-wavs/down.wav", help="Path to WAV file to test"
    )
    parser.add_argument(
        "--wav_dir", default="bcm-wavs", help="Directory containing WAV files to test"
    )
    parser.add_argument(
        "--model_path",
        default="trained_models/kws_ref_model.tflite",
        help="Path to model file",
    )
    parser.add_argument(
        "--is_tflite", action="store_true", default=True, help="Flag for TFLite model"
    )
    parser.add_argument(
        "--test_all", action="store_true", help="Test all WAV files in the directory"
    )

    # Parse our arguments
    test_args = parser.parse_args()

    # Update the flags with our test arguments
    flags.wav_file = test_args.wav_file
    flags.is_tflite = test_args.is_tflite

    # Update model paths based on provided model_path
    if test_args.model_path.endswith(".tflite"):
        flags.tfl_file_name = test_args.model_path
        flags.is_tflite = True
    else:
        flags.model_init_path = test_args.model_path

    # Load model
    print(f"\n\nLoading model from {test_args.model_path}")
    is_quantized = check_if_quantized(test_args.model_path)
    print(f"Is the model quantized? {'Yes' if is_quantized else 'No'}\n\n")
    try:
        if flags.is_tflite:
            # For TFLite model, we'll load it during inference
            model = None
        else:
            model = tf.keras.models.load_model(test_args.model_path)
            model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Test all WAV files or a single WAV file
    if test_args.test_all:
        print(f"Testing all WAV files in directory: {test_args.wav_dir}")
        test_all_wavs(model, test_args.wav_dir, flags)
    else:
        # Test single WAV file
        print(f"Testing WAV file: {flags.wav_file}")
        result = test_wav_file(model, flags.wav_file, flags)

        if result is None:
            print("Prediction failed")
            return

        # Display results
        class_id, label, confidence, all_scores = result
        ground_truth = get_ground_truth_from_filename(flags.wav_file)

        print("\nPrediction Results:")
        print(f"Ground Truth: {ground_truth}")
        print(f"Predicted Class: {class_id}")
        print(f"Predicted Label: {label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Correct: {'Yes' if label.lower() == ground_truth.lower() else 'No'}")

        print("\nAll Class Probabilities:")
        for i, score in enumerate(all_scores):
            print(f"  {WORD_LABELS[i]}: {score:.4f}")


if __name__ == "__main__":
    main()
