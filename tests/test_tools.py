import pytest
from unittest.mock import patch

import gradio_tools
from gradio_tools import GradioTool

try:
    import gradio as gr
    GRADIO_INSTALLED = True
except:
    GRADIO_INSTALLED = False


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


@pytest.mark.skipif(not GRADIO_INSTALLED, reason="Gradio not installed")
@pytest.mark.parametrize("tool_class", GradioTool.__subclasses__())
def test_input_output(tool_class):
    if tool_class.__name__ == "BarkTextToSpeechTool":
        inp = [gr.Textbox()]
        output = [gr.Audio()]
    elif tool_class.__name__ == "ClipInterrogatorTool":
        inp = [gr.Image()]
        output = [gr.Textbox()]
    elif tool_class.__name__ == "DocQueryDocumentAnsweringTool":
        inp = [gr.Image(), gr.Textbox()]
        output = [gr.Textbox()]
    elif tool_class.__name__ == "ImageCaptioningTool":
        inp = [gr.Image()]
        output = [gr.Textbox()]
    elif tool_class.__name__ == "ImageToMusicTool":
        inp = [gr.Image()]
        output = [gr.Audio()]
    elif tool_class.__name__ == "StableDiffusionPromptGeneratorTool":
        inp = [gr.Textbox()]
        output = [gr.Textbox()]
    elif tool_class.__name__ == "SAMImageSegmentationTool":
        inp = [gr.Number(), gr.Number(), gr.Number(), gr.Image(), gr.Textbox()]
        output = [gr.Image()]
    elif tool_class.__name__ == "StableDiffusionTool":
        inp = [gr.Textbox()]
        output = [gr.Image()]
    elif tool_class.__name__ == "TextToVideoTool":
        inp = [gr.Textbox()]
        output = [gr.Video()]
    elif tool_class.__name__ == "WhisperAudioTranscriptionTool":
        inp = [gr.Audio()]
        output = [gr.Textbox()]
    else:
        raise ValueError(f"Test does not have a case for: {tool_class.__name__}")
    
    tool = tool_class()
    assert [t.__class__ for t in tool.block_input()] == [t.__class__ for t in inp]
    assert [t.__class__ for t in tool.block_output()] == [t.__class__ for t in output]

