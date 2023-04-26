import pytest
from unittest.mock import patch

import gradio_tools
from gradio_tools import GradioTool


@pytest.mark.parametrize("tool_class", GradioTool.__subclasses__())
@patch("gradio_client.Client.duplicate")
def test_duplicate(mock_duplicate, tool_class):
    tool_class(duplicate=True, hf_token="dafsdf")
    mock_duplicate.assert_called_once()


@pytest.mark.parametrize("tool_class", GradioTool.__subclasses__())
@patch("gradio_client.Client.duplicate")
def test_dont_duplicate(mock_duplicate, tool_class):
    tool_class(duplicate=False)
    mock_duplicate.assert_not_called()


@pytest.mark.parametrize("tool_class", GradioTool.__subclasses__())
def test_all_listed_in_init(tool_class):
    assert tool_class.__name__ in gradio_tools.__all__