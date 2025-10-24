#!/bin/bash

if [[ "$1" == "--help" ]]; then
    echo "How to: $0 [option]"
    echo "  --help     Instructions"
    echo "  phoneme    Preprocess for phoneme-based model"
    echo "  normal     Preprocess for normal model (default)"
    exit 0
fi

set -e  

mkdir -p dataset
cd dataset

pip install gdown librosa speechbrain jiwer
pip install https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20250807+cuda12.8.torch2.8.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
pip install git+https://github.com/lhotse-speech/lhotse

gdown 19CV4WZgYez-i2oHV2r9maJofjNqcTX4o 
gdown 1v75mLO-TVfPXe27o54JMlXD5cQ81eaVG 
gdown 1YgTF-NbHuweHWr2LahS_X9j--laGDnIK

unzip -o voices.zip

cd /
if [[ "$1" == "phoneme" ]]; then
    echo "Preprocessing for phoneme-based model"
    python workspace/Zipformer/utils/phoneme_construct.py
elif [[ "$1" == "char" ]]; then
    echo "Preprocessing for normal model"
    python workspace/Zipformer/utils/char_construct.py
else
    echo "Preprocessing for normal model"
    python workspace/Zipformer/utils/construct.py
fi
mkdir workspace/Zipformer/saves