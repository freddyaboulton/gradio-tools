# Gradio Tool: Gradio 🤝 LangChain Agents

Any [Gradio](https://github.com/gradio-app/gradio) application at the tips of your LLM's fingers 🦾

[gradio_tool](https://github.com/freddyaboulton/gradio-tool) can turn any [Gradio](https://github.com/gradio-app/gradio) application into a [tool](https://python.langchain.com/en/latest/modules/agents/tools.html) that a [LangChain agent](https://docs.langchain.com/docs/components/agents/agent) can use to complete its task.

## Example Usage

Simply import the desired tools from `gradio_tool` (or create your own!) and pass to `initialize_agent` from LangChain.

In this example, we use some pre-built tools to generate images, caption them, and create a music clip to match its artistic style!

Read the [How It Works](#how-it-works) section to learn how to create your own tools! We welcome any new tools to the library!

```python
os.environ["OPENAI_API_KEY"] = "<Secret Key>"

from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import os
from gradio_tool.tool import StableDiffusionTool, ImageCaptioningTool, ImageToMusicTool
from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
tools = [StableDiffusionTool(), ImageCaptioningTool(), ImageToMusicTool()]


agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)
output = agent.run(input=("I would please like a photo of a dog riding a skateboard. "
                          "Please caption this image and create a song for it."))
```


## How it works

The core abstraction is the `GradioTool`, which lets you define a new tool for your LLM as long as you implement a standard interface:

```python
class GradioTool(BaseTool):
    name = "GradioTool"
    description: str
    src: str
    
    @abstractmethod
    def create_job(self, query: str) -> Job:
        pass

    @abstractmethod
    def postprocess(self, output: Tuple[Any]) -> str:
        pass
```

The requirements are:
1. The name for your tool
2. The description for your tool. This is crucial! Agents decide which tool to use based on their description. Be precise and be sure to inclue example of what the input and the output of the tool should look like.
3. The url or space id, e.g. `freddyaboulton/calculator`, of the Gradio application. Based on this value, `gradio_tool` will create a [gradio client](https://github.com/gradio-app/gradio/blob/main/client/python/README.md) instance to query the upstream application via API. Be sure to click the link and learn more about the gradio client library if you are not familiar with it.
4. create_job - Given a string, this method should parse that string and return a job from the client. Most times, this is as simple as passing the string to the `submit` function of the client. More info on creating jobs [here](https://github.com/gradio-app/gradio/blob/main/client/python/README.md#making-a-prediction)
5. postprocess - Given the result of the job, convert it to a string the LLM can display to the user.

And that's it!

## Pre-built tools

gradio_tool comes with a set of pre-built tools you can leverage immediately! These are

1. StableDiffusionTool - Generate an image from a given prompt using the open source stable diffusion demo hosted on [HuggingFace spaces](https://huggingface.co/spaces/stabilityai/stable-diffusion)
2. ImageCaptionTool - Caption an image by providing a filepath based on Niels Rogge's [HuggingFace Space](https://huggingface.co/spaces/nielsr/comparing-captioning-models)
3. ImageToMusicTool - Create an audio clip that matches the style of a given image file based on Sylvain Filoni's [HuggingFace Space](https://huggingface.co/spaces/fffiloni/img-to-music)

## Appendix

### What are agents?

A [LangChain agent](https://docs.langchain.com/docs/components/agents/agent) is a Large Language Model (LLM) that takes user input and reports an output based on using one of many tools at its disposal.

### What is Gradio?
[Gradio](https://github.com/gradio-app/gradio) is the defacto standard tool for building Machine Learning Web Applications and sharing them with the world - all with just python! 🐍
