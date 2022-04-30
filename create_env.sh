#!/bin/bash

set -eu

PROJECT_ROOT_DIR=$(dirname "$0")

# set up conda environment
conda env create --file "${PROJECT_ROOT_DIR}"/env.yaml --prefix "${PROJECT_ROOT_DIR}"/env

# install package into conda environment
"${PROJECT_ROOT_DIR}"/env/bin/pip install -e "${PROJECT_ROOT_DIR}"
