import pytest
from unittest.mock import patch

from gradio_tools import SAMImageSegmentationTool


@patch("gradio_client.Client.submit")
def test_input_parsing(mock_submit):
    tool = SAMImageSegmentationTool()
    tool.create_job("my_image.png| a red horse|0.9|0.8|0.9")
    mock_submit.assert_called_with(0.9,0.8,0.9,"my_image.png", "a red horse", api_name="/predict")


@patch("gradio_client.Client.submit")
def test_raise_error(mock_submit):
    tool = SAMImageSegmentationTool()
    with pytest.raises(ValueError,
                       match="Not enough arguments passed to the SAMImageSegmentationTool!"):
        tool.create_job("my_image.png| a red horse|")