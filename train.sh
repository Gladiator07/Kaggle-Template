#!/bin/bash
cd src/
python train.py --config ../configs/cfg_baseline.py \
                --config.version 1 \
                --config.notes "baseline experiment"