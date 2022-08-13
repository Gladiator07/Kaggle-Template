#!/bin/bash
cd src/
python train.py --config ../configs/cfg_baseline.py \
                --fold 0 \
                --config.version 0 \
                --config.notes "baseline experiment"