import os
import time
from abc import abstractmethod
from functools import wraps
from typing import Any, Tuple, Union

import gradio_client as grc
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
    def __init__(self, name: str, description: str, src: str) -> None:
        self.name = name
        self.description = description
        self.src = src
        self.client = grc.Client(self.src)
        self.description += (
            "If the output is the string QUEUE_FULL you "
            "should try this action again and never stop until you "
            "don't see this message."
        )
        self._block = None

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
    ) -> None:
        super().__init__(name, description, src)

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
    ) -> None:
        super().__init__(name, description, src)

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
    ) -> None:
        super().__init__(name, description, src)

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
