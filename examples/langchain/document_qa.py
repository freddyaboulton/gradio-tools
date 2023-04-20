import os
import pathlib

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY must be set")

from langchain.agents import initialize_agent, load_tools
from langchain.llms import OpenAI
from gradio_tools import DocQueryDocumentAnsweringTool

from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
tools = [DocQueryDocumentAnsweringTool().langchain]

IMG_PATH = pathlib.Path(__file__).parent / "florida-drivers-license.jpeg"

agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)
output = agent.run(input=f"What is the date of birth the driver in {IMG_PATH}?")
output = agent.run(input=f"What is the current date?")
output = agent.run(input=f"Using the current date, what is the age of the driver? Explain your reasoning.")
output = agent.run(input=f"What is the driver's license number?")