from abc import abstractmethod
from pydantic import Field, root_validator, validator

from langchain.tools.base import BaseTool

import gradio_client as grc
from gradio_client.client import Job
from gradio_client.utils import QueueError
import time
import os
from typing import Tuple, Any, Union


class GradioTool(BaseTool):
    name = "GradioTool"
    description: str
    src: str

    class Config:
        """Configuration for this pydantic object."""
        extra = 'allow'
    
    @abstractmethod
    def create_job(self, query: str) -> Job:
        pass

    @abstractmethod
    def postprocess(self, output: Tuple[Any]) -> str:
        pass

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.client: grc.Client = grc.Client(self.src)
    
    @validator("description", pre=True, always=True)
    def set_description(cls, value):
        return value + ("If the output is the string QUEUE_FULL you "
                                  "should try this action again and never stop until you "
                                  "don't see this message.")

    def _run(self, query: str):
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

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError(f"{self.name} does not support async")


class StableDiffusionTool(GradioTool):
    """Tool for calling stable diffusion from llm"""

    name = "StableDiffusion"
    description = ("An image generator. Use this to generate images based on "
                   "text input. Input should be a description of what the image should "
                   "look like. The output will be a path to an image file.")
    src = "stabilityai/stable-diffusion"

    def create_job(self, query: str) -> Job:
        return self.client.submit(query, "", 9, fn_index=1)
    
    def postprocess(self, output: Tuple[Any]) -> str:
        return [os.path.join(output, i) for i in os.listdir(output) if not i.endswith("json")][0]
    


class ImageCaptioningTool(GradioTool):
    """Tool for captioning images."""

    name = "ImageCaptioner"
    description = ("An image captioner. Use this to create a caption for an image. "
                   "Input will be a path to an image file. "
                   "The output will be a caption of that image. ")
    src = "nielsr/comparing-captioning-models"

    def create_job(self, query: str) -> Job:
        return self.client.submit(query.strip("'"), fn_index=0)
    
    def postprocess(self, output: Tuple[Any]) -> str:
        return output[1]


class ImageToMusicTool(GradioTool):
    name = "ImagetoMusic"
    description = """A tool for creating music from images. Use this tool to create a musical track from an image. Input will be a path to an image file.
    The output will be an audio file generated from that image.
    """
    src = "fffiloni/img-to-music"

    def create_job(self, query: str) -> Job:
        return self.client.submit(query.strip("'"), 15, "medium", "loop", None, fn_index=0)

    def postprocess(self, output: Tuple[Any]) -> str:
        return output[1]
