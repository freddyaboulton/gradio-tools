from __future__ import annotations

import os
import time
from abc import abstractmethod
from functools import wraps
from typing import Any, Tuple, Union

import gradio_client as grc
import huggingface_hub
from gradio_client.client import Job
from gradio_client.utils import QueueError

try:
    import gradio as gr

    GRADIO_INSTALLED = True
except (ModuleNotFoundError, ImportError):
    GRADIO_INSTALLED = False

try:
    import langchain as lc

    LANGCHAIN_INSTALLED = True
except (ModuleNotFoundError, ImportError):
    LANGCHAIN_INSTALLED = False


def raises_without_gradio(f):
    @wraps(f)
    def _f(*args, **kwargs):
        if not GRADIO_INSTALLED:
            raise ModuleNotFoundError(f"gradio must be installed to call {f.__name__}")

    return _f


class GradioTool:
    def __init__(
        self, name: str, description: str, src: str, hf_token: str | None = None
    ) -> None:
        self.name = name
        self.description = description
        if hf_token and self._is_space(src):
            self.client = grc.Client.duplicate(from_id=src, hf_token=hf_token)
            self.src = self.client.space_id
        else:
            self.src = src
            self.client = grc.Client(self.src)
        self.description += (
            "If the output is the string QUEUE_FULL you "
            "should try this action again and never stop until you "
            "don't see this message."
        )
        self._block = None

    @staticmethod
    def _is_space(src: str) -> bool:
        try:
            huggingface_hub.get_space_runtime(src)
            return True
        except huggingface_hub.hf_api.RepositoryNotFoundError:
            return False

    @abstractmethod
    def create_job(self, query: str) -> Job:
        pass

    @abstractmethod
    def postprocess(self, output: Union[Tuple[Any], Any]) -> str:
        pass

    def run(self, query: str):
        job = self.create_job(query)
        while not job.done():
            status = job.status()
            print(f"\nJob Status: {str(status.code)} eta: {status.eta}")
            time.sleep(30)
        try:
            output = self.postprocess(job.result())
        except QueueError:
            output = "QUEUE_FULL"
        return output

    # Optional gradio functionalities
    def _block_input(self) -> "gr.components.Component":
        return gr.Textbox()

    def _block_output(self) -> "gr.components.Component":
        return gr.Textbox()

    @raises_without_gradio
    def block_input(self) -> "gr.components.Component":
        return self._block_input()

    @raises_without_gradio
    def block_output(self) -> "gr.components.Component":
        return self._block_output()

    @raises_without_gradio
    def block(self):
        """Get the gradio Blocks of this tool for visualization."""
        if not self._block:
            self._block = gr.load(name=self.src, src="spaces")
        return self._block

    # Optional langchain functionalities
    @property
    def langchain(self) -> "langchain.agents.Tool":
        if not LANGCHAIN_INSTALLED:
            raise ModuleNotFoundError(
                "langchain must be installed to access langchain tool"
            )

        return lc.agents.Tool(
            name=self.name, func=self.run, description=self.description
        )

    def __repr__(self) -> str:
        return f"GradioTool(name={self.name}, src={self.src})"


class StableDiffusionTool(GradioTool):
    """Tool for calling stable diffusion from llm"""

    def __init__(
        self,
        name="StableDiffusion",
        description=(
            "An image generator. Use this to generate images based on "
            "text input. Input should be a description of what the image should "
            "look like. The output will be a path to an image file."
        ),
        src="gradio-client-demos/stable-diffusion",
        hf_token=None,
    ) -> None:
        super().__init__(name, description, src, hf_token)

    def create_job(self, query: str) -> Job:
        return self.client.submit(query, "", 9, fn_index=1)

    def postprocess(self, output: Union[Tuple[Any], Any]) -> str:
        assert isinstance(output, str)
        return [
            os.path.join(output, i)
            for i in os.listdir(output)
            if not i.endswith("json")
        ][0]

    def _block_input(self) -> "gr.components.Component":
        return gr.Textbox()

    def _block_output(self) -> "gr.components.Component":
        return gr.Image()


class ImageCaptioningTool(GradioTool):
    """Tool for captioning images."""

    def __init__(
        self,
        name="ImageCaptioner",
        description=(
            "An image captioner. Use this to create a caption for an image. "
            "Input will be a path to an image file. "
            "The output will be a caption of that image."
        ),
        src="gradio-client-demos/comparing-captioning-models",
        hf_token=None,
    ) -> None:
        super().__init__(name, description, src, hf_token)

    def create_job(self, query: str) -> Job:
        return self.client.submit(query.strip("'"), fn_index=0)

    def postprocess(self, output: Union[Tuple[Any], Any]) -> str:
        return output[1]  # type: ignore

    def _block_input(self) -> "gr.components.Component":
        return gr.Image()

    def _block_output(self) -> "gr.components.Component":
        return gr.Textbox()


class ImageToMusicTool(GradioTool):
    def __init__(
        self,
        name="ImagetoMusic",
        description=(
            "A tool for creating music from images. Use this tool to create a musical "
            "track from an image. Input will be a path to an image file. "
            "The output will be an audio file generated from that image."
        ),
        src="fffiloni/img-to-music",
        hf_token=None,
    ) -> None:
        super().__init__(name, description, src, hf_token)

    def create_job(self, query: str) -> Job:
        return self.client.submit(
            query.strip("'"), 15, "medium", "loop", None, fn_index=0
        )

    def postprocess(self, output: Union[Tuple[Any], Any]) -> str:
        return output[1]  # type: ignore

    def _block_input(self) -> "gr.components.Component":
        return gr.Image()

    def _block_output(self) -> "gr.components.Component":
        return gr.Audio()


class WhisperTool(GradioTool):
    def __init__(
        self,
        name="Whisper",
        description=(
            "A tool for transcribing audio. Use this tool to transcribe an audio file. "
            "track from an image. Input will be a path to an audio file. "
            "The output will the text transcript of that file."
        ),
        src="abidlabs/whisper-large-v2",
        hf_token=None,
    ) -> None:
        super().__init__(name, description, src, hf_token)

    def create_job(self, query: str) -> Job:
        return self.client.submit(query, api_name="/predict")

    def postprocess(self, output: str) -> str:
        return output

    def _block_input(self) -> "gr.components.Component":
        return gr.Audio()

    def _block_output(self) -> "gr.components.Component":
        return gr.Textbox()


class StableDiffusionPromptGeneratorTool(GradioTool):
    def __init__(
        self,
        name="StableDiffusionPromptGenerator",
        description=(
            "Use this tool to improve a prompt for stable diffusion and other image generators "
            "This tool will refine your prompt to include key words and phrases that make "
            "stable diffusion perform better. The input is a prompt text string "
            "and the output is a prompt text string"
        ),
        src="microsoft/Promptist",
        hf_token=None,
    ) -> None:
        super().__init__(name, description, src, hf_token)

    def create_job(self, query: str) -> Job:
        return self.client.submit(query, api_name="/predict")

    def postprocess(self, output: str) -> str:
        return output


class ClipInterrogatorTool(GradioTool):
    def __init__(
        self,
        name="ClipInterrogator",
        description=(
            "A tool for reverse engineering a prompt from a source image. "
            "Use this tool to create a prompt for StableDiffusion that matches the "
            "input image. The imput is a path to an image. The output is a text string."
        ),
        src="pharma/CLIP-Interrogator",
        hf_token=None,
    ) -> None:
        super().__init__(name, description, src, hf_token)

    def create_job(self, query: str) -> Job:
        return self.client.submit(
            query, "ViT-L (best for Stable Diffusion 1.*)", "best", fn_index=3
        )

    def postprocess(self, output: str) -> str:
        return output

    def _block_input(self) -> "gr.components.Component":
        return gr.Image()


class TextToVideoTool(GradioTool):
    def __init__(
        self,
        name="TextToVideo",
        description=(
            "A tool for creating videos from text."
            "Use this tool to create videos from text prompts. "
            "Input will be a text prompt describing a video scene. "
            "The output will be a path to a video file."
        ),
        src="damo-vilab/modelscope-text-to-video-synthesis",
        hf_token=None,
    ) -> None:
        super().__init__(name, description, src, hf_token)

    def create_job(self, query: str) -> Job:
        return self.client.submit(query, -1, 16, 25, fn_index=1)

    def postprocess(self, output: str) -> str:
        return output

    def _block_output(self) -> "gr.components.Component":
        return gr.Video()
