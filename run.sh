#!/bin/bash

conda activate tf

python TensorFlowASR/examples/demonstration/conformer.py \
    --config ../config3.yml \
    --saved ../latest2.h5 \
    --subwords ../conformer.subwords \
    --output_dir ../pretrained \
    ${AUDIO_PATH}
