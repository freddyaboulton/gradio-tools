set -e
set -x

pyright gradio_tools
ruff gradio_tools
black gradio_tools --check
isort gradio_tools --check-only