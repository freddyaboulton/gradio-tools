#!/bin/sh -e
set -x

ruff gradio_tools --fix
black gradio_tools
isort gradio_tools