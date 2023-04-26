# Gradio Tools: Gradio 🤝 LLM Agents

There are many 1000s of [Gradio](https://github.com/gradio-app/gradio) apps on [Hugging Face Spaces](https://huggingface.co/spaces). This library puts them at the tips of your LLM's fingers 🦾

Specifically, [`gradio-tools`](https://pypi.org/project/gradio-tools/) is a Python library for converting Gradio apps into [tools](https://python.langchain.com/en/latest/modules/agents/tools.html) that can be leveraged by a large language model (LLM)-based agent to complete its task. For example, an LLM could use a Gradio tool to transcribe a voice recording it finds online and then summarize it for you. Or it could use a different Gradio tool to apply OCR to a document on your Google Drive and then answer questions about it.

Currently supported libraries for agents are:
- [LangChain](https://docs.langchain.com/docs/components/agents/agent)
- [MiniChain](https://github.com/srush/MiniChain/tree/main)

`gradio-tools` comes with a set of pre-built tools you can leverage immediately! These include:

1. StableDiffusionTool - Generate an image from a given prompt using the open source stable diffusion demo hosted on [HuggingFace spaces](https://huggingface.co/spaces/stabilityai/stable-diffusion)
2. ImageCaptionTool - Caption an image by providing a filepath based on Niels Rogge's [HuggingFace Space](https://huggingface.co/spaces/nielsr/comparing-captioning-models)
3. ImageToMusicTool - Create an audio clip that matches the style of a given image file based on Sylvain Filoni's [HuggingFace Space](https://huggingface.co/spaces/fffiloni/img-to-music)
4. StableDiffusionPromptGeneratorTool - Use this tool to improve a prompt for stable diffusion and other image generators based on this [HuggingFace Space](https://huggingface.co/spaces/microsoft/Promptist)
5. TextToVideoTool - A tool for creating short videos from text. Based on this [HuggingFace Space](https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis)
6. WhisperAudioTranscriptionTool - A tool for transcribing audio with Whisper. Based on this [HuggingFace Space](https://huggingface.co/spaces/abidlabs/whisper)
7. ClipInterrogatorTool - A tool for reverse engineering a prompt from a source image. Based on this [HuggingFace Space](https://huggingface.co/spaces/pharma/CLIP-Interrogator)
8. DocQueryDocumentAnsweringTool - A tool for answering questions about a document from the from the image of the document. Based on this [HuggingFace Space](https://huggingface.co/spaces/abidlabs/docquery)
9. BarkTextToSpeechTool - A tool for text-to-speech. Based on this [HuggingFace Space](https://huggingface.co/spaces/suno/bark)

We welcome more contributions!

## Example Usage

Simply import the desired tools from `gradio_tools` (or create your own!) and pass to `initialize_agent` from LangChain.

In this example, we use some pre-built tools to generate images, caption them, and create a video!

Read the [How It Works](#how-it-works) section to learn how to create your own tools! We welcome any new tools to the library!

```python
from gradio_tools import (StableDiffusionTool, ImageCaptioningTool, StableDiffusionPromptGeneratorTool,
                          TextToVideoTool)

from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")

tools = [StableDiffusionTool().langchain, ImageCaptioningTool().langchain,
         StableDiffusionPromptGeneratorTool().langchain, TextToVideoTool().langchain]


agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)
output = agent.run(input=("Please create a photo of a dog riding a skateboard "
                          "but improve my prompt prior to using an image generator."
                          "Please caption the generated image and create a video for it using the improved prompt."))
```

https://user-images.githubusercontent.com/41651716/233391796-49c3f762-bd58-41d7-b399-629e3c1661b2.mp4


See the `/examples` directory for more complete code examples. 

## How it works

The core abstraction is the `GradioTool`, which lets you define a new tool for your LLM as long as you implement a standard interface:

```python
class GradioTool(BaseTool):

    def __init__(self, name: str, description: str, src: str) -> None:

    @abstractmethod
    def create_job(self, query: str) -> Job:
        pass

    @abstractmethod
    def postprocess(self, output: Tuple[Any] | Any) -> str:
        pass
```

The requirements are:
1. The name for your tool
2. The description for your tool. This is crucial! Agents decide which tool to use based on their description. Be precise and be sure to inclue example of what the input and the output of the tool should look like.
3. The url or space id, e.g. `freddyaboulton/calculator`, of the Gradio application. Based on this value, `gradio_tools` will create a [gradio client](https://github.com/gradio-app/gradio/blob/main/client/python/README.md) instance to query the upstream application via API. Be sure to click the link and learn more about the gradio client library if you are not familiar with it.
4. create_job - Given a string, this method should parse that string and return a job from the client. Most times, this is as simple as passing the string to the `submit` function of the client. More info on creating jobs [here](https://github.com/gradio-app/gradio/blob/main/client/python/README.md#making-a-prediction)
5. postprocess - Given the result of the job, convert it to a string the LLM can display to the user.
6. *Optional* - Some libraries, e.g. [MiniChain](https://github.com/srush/MiniChain/tree/main), may need some info about the underlying gradio input and output types used by the tool. By default, this will return gr.Textbox() but 
if you'd like to provide more accurate info, implement the `_block_input(self, gr)` and `_block_output(self, gr)` methods of the tool. The `gr` variable is the gradio module (the result of `import gradio as gr`). It will be
automatically imported by the `GradiTool` parent class and passed to the `_block_input` and `_block_output` methods.

And that's it!



## Appendix

### What are agents?

A [LangChain agent](https://docs.langchain.com/docs/components/agents/agent) is a Large Language Model (LLM) that takes user input and reports an output based on using one of many tools at its disposal.

### What is Gradio?
[Gradio](https://github.com/gradio-app/gradio) is the defacto standard tool for building Machine Learning Web Applications and sharing them with the world - all with just python! 🐍
