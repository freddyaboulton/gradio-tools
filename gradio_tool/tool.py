from abc import abstractmethod
from pydantic import Field

from langchain.tools.base import BaseTool

import gradio_client as grc
from gradio_client.client import Job
from gradio_client.utils import QueueError
import time
import os
from typing import Tuple, Any


class GradioTool(BaseTool):
    name = "GradioTool"
    description: str
    client: grc.Client
    
    @abstractmethod
    def create_job(self, query: str) -> Job:
        pass

    @abstractmethod
    def postprocess(self, output: Tuple[Any]) -> str:
        pass

    def _run(self, query: str):
        job = self.create_job(query)
        while not job.done():
            status = job.status()
            print(f"\nJob Status: {str(status.code)} eta: {status.eta}")
            time.sleep(20)
        try:
            output = self.postprocess(job.result())
        except QueueError:
            output = "QUEUE_FULL"
        return output

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError(f"{self.name} does not support async")


class StableDiffusionTool(GradioTool):
    """Tool for calling stable diffusion from llm"""

    name = "StableDiffusion"
    description = ("An image generator. Use this to generate images based on "
                   "text input. Input should be a description of what the image should "
                   "look like. The output will be a path to an image file. "
                   "If the output is the string QUEUE_FULL you should try this action again and never stop until you don't see this message.")
    client: grc.Client = Field(default_factory=lambda: grc.Client(src="stabilityai/stable-diffusion"))

    def create_job(self, query: str) -> Job:
        return self.client.predict(query, "", 9, fn_index=1)
    
    def postprocess(self, output: Tuple[Any]) -> str:
        return [os.path.join(output, i) for i in os.listdir(output) if not i.endswith("json")][0]
    


class ImageCaptioningTool(GradioTool):
    """Tool for captioning images."""

    name = "ImageCaptioner"
    description = ("An image captioner. Use this to create a caption for an image. "
                   "Input will be a path to an image file. "
                   "The output will be a caption of that image. "
                   "If the output is the string QUEUE_FULL you should try this action again.")
    client: grc.Client = Field(default_factory=lambda: grc.Client(src="nielsr/comparing-captioning-models"))

    def create_job(self, query: str) -> Job:
        return self.client.predict(query.strip("'"), fn_index=0)
    
    def postprocess(self, output: Tuple[Any]) -> str:
        return output[1]


class ImageToMusicTool(GradioTool):
    name = "ImagetoMusic"
    description = """A tool for creating music from images. Use this tool to create a musical track from an image. Input will be a path to an image file.
    The output will be an audio file generated from that image. If the output is the string QUEUE_FULL you should try this action again.
    """
    client: grc.Client = Field(default_factory=lambda: grc.Client(src="fffiloni/img-to-music"))

    def create_job(self, query: str) -> Job:
        return self.client.predict(query.strip("'"), 15, "medium", "loop", None, fn_index=0)

    def postprocess(self, output: Tuple[Any]) -> str:
        return output[1]
