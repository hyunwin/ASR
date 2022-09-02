#!/bin/bash

git clone https://github.com/TensorSpeech/TensorFlowASR.git

cd TensorFlowASR

conda create -n tfasr python==3.8 black flake8 tensorflow -y

conda activate tfasr

pip install -e ".[tf2.6]"
