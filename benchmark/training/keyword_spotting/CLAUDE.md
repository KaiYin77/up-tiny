# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Keyword Spotting (KWS) training component of the MLPerf Tiny benchmark suite. It implements a deep neural network for recognizing 12 keywords: "Down", "Go", "Left", "No", "Off", "On", "Right", "Stop", "Up", "Yes", "Silence", and "Unknown". The model uses the Google Speech Commands v2 dataset and deploys to microcontrollers using TensorFlow Lite for Microcontrollers (TFLM).

## Common Development Commands

### Training
- `python train.py` - Train the KWS model (downloads dataset automatically)
- `./build_ref.sh` - Complete training pipeline (train → quantize → evaluate)

### Model Conversion and Evaluation
- `python quantize.py` - Convert trained model to quantized TFLite format
- `python eval_quantized_model.py` - Evaluate quantized model accuracy
- `./tflm2cc trained_models/kws_model.tflite` - Convert TFLite model to C++ source

### Testing
- `./test_wav2kws_tflite.sh` - Test model with WAV files in bcm-wavs/ directory
- `./test_wav2kws_tflite.sh [file.wav]` - Test specific WAV file

### Binary Dataset Generation
- `./make_all_bin_files.sh` - Generate all binary test files (MFCC, LFBE, time-domain)
- `python make_bin_files.py --feature_type=[mfcc|lfbe|td_samples] --bin_file_path=PATH` - Generate specific feature type

## Architecture

### Core Components
- **keras_model.py**: Model architecture definitions (DS-CNN, CNN variants)
- **get_dataset.py**: Dataset loading, preprocessing, and TensorFlow dataset creation
- **kws_util.py**: Command-line argument parsing and utility functions
- **train.py**: Main training script with data loading and model fitting
- **quantize.py**: TFLite quantization with calibration dataset
- **wav2kws_tflite.py**: WAV file testing and inference

### Feature Types
- **MFCC**: Mel-Frequency Cepstral Coefficients (49x10, INT8)
- **LFBE**: Log Filter-bank Energies (41x40, UINT8) 
- **td_samples**: Raw time-domain waveform (16000x1, INT16)

### Model Pipeline
1. **Data**: Google Speech Commands v2 → TensorFlow Dataset
2. **Training**: Keras model training with data augmentation
3. **Quantization**: INT8 post-training quantization using calibration set
4. **Deployment**: C++ code generation for microcontroller deployment

## File Structure

- `trained_models/` - Contains trained models (.h5, .tflite formats)
- `bcm-wavs/` - Test WAV files for validation
- `quant_cal_idxs.txt` - Calibration dataset indices for quantization
- Reference implementation at `../../reference_submissions/keyword_spotting/`

## Key Parameters

- 12 classes (word labels)
- 16kHz sample rate, 1-second clips
- Default: 36 training epochs
- Batch size and data augmentation configurable via command line
- Multiple model architectures available (DS-CNN, CNN variants)