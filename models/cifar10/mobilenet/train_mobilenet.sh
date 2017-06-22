#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --gpu=0 \
    --solver=examples/cifar10/cifar10_mobilenet_solver.prototxt \
        2>&1 | tee -a examples/cifar10/log/mobilenet.log $@
