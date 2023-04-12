import os

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY must be set")

from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from gradio_tools.tools import StableDiffusionTool, ImageCaptioningTool, ImageToMusicTool
from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
tools = [StableDiffusionTool().langchain, ImageCaptioningTool().langchain]


agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)
breakpoint()
output = agent.run(input=("I would please like a photo of a dog riding a skateboard. "
                          "Please caption this image and create a song for it."))
