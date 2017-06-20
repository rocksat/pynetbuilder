#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_mobilenet_solver.prototxt \
    --gpu=0,1 $@

