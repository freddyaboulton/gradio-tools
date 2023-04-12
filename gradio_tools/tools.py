from abc import abstractmethod

from langchain.agents import Tool

import gradio as gr
import gradio_client as grc
from gradio_client.client import Job
from gradio_client.utils import QueueError
import time
import os
from typing import Tuple, Any


class GradioTool:

    def __init__(self, name: str, description: str, src: str) -> None:
        self.name = name
        self.description = description
        self.src = src
        self.client = grc.Client(self.src)
        self.description += ("If the output is the string QUEUE_FULL you "
                        "should try this action again and never stop until you "
                        "don't see this message.")
        self._langchain = Tool(name=self.name, func=self.run, description=self.description)

    @abstractmethod
    def create_job(self, query: str) -> Job:
        pass

    @abstractmethod
    def postprocess(self, output: Tuple[Any]) -> str:
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
    
    def block(self,):
        """Get the gradio Blocks corresponding to this tool - useful for visualization."""
        return gr.load(self.src)
    
    @property
    def langchain(self):
        return self._langchain
    

class StableDiffusionTool(GradioTool):
    """Tool for calling stable diffusion from llm"""

    def __init__(self, name="StableDiffusion",
                 description=("An image generator. Use this to generate images based on "
                   "text input. Input should be a description of what the image should "
                   "look like. The output will be a path to an image file."),
                   src= "gradio-client-demos/stable-diffusion") -> None:
        super().__init__(name, description, src)


    def create_job(self, query: str) -> Job:
        return self.client.submit(query, "", 9, fn_index=1)
    
    def postprocess(self, output: Tuple[Any]) -> str:
        return [os.path.join(output, i) for i in os.listdir(output) if not i.endswith("json")][0]


class ImageCaptioningTool(GradioTool):
    """Tool for captioning images."""

    def __init__(self, name= "ImageCaptioner",
                 description=("An image captioner. Use this to create a caption for an image. "
                   "Input will be a path to an image file. "
                   "The output will be a caption of that image."),
                   src="gradio-client-demos/comparing-captioning-models") -> None:
        super().__init__(name, description, src)

    def create_job(self, query: str) -> Job:
        return self.client.submit(query.strip("'"), fn_index=0)
    
    def postprocess(self, output: Tuple[Any]) -> str:
        return output[1]


class ImageToMusicTool(GradioTool):

    def __init__(self, name="ImagetoMusic",
                 description=("A tool for creating music from images. Use this tool to create a musical "
                              "track from an image. Input will be a path to an image file. "
                              "The output will be an audio file generated from that image."),
                src="fffiloni/img-to-music") -> None:
        super().__init__(name, description, src)

    def create_job(self, query: str) -> Job:
        return self.client.submit(query.strip("'"), 15, "medium", "loop", None, fn_index=0)

    def postprocess(self, output: Tuple[Any]) -> str:
        return output[1]
