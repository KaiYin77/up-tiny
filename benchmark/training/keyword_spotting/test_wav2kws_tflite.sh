#!/bin/bash
# Script to test KWS model with 32-bit WAV files

# Define colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Testing KWS model with 32-bit WAV files...${NC}"

# If a specific file is provided, test that one
if [ "$1" != "" ]; then
    echo -e "Testing specific file: $1"
    python wav2kws_tflite.py --wav_file "$1"
else
    # Otherwise test all files in the bcm-wavs directory
    echo -e "Testing all WAV files in bcm-wavs directory"
    python wav2kws_tflite.py --test_all
fi

echo -e "${GREEN}Test completed!${NC}"
