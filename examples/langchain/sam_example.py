import os
import pathlib


if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY must be set")

from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from gradio_tools.tools import SAMImageSegmentationTool

from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
tools = [SAMImageSegmentationTool().langchain]

waldo_1 = pathlib.Path(__file__).parent / "waldo.jpeg"
waldo_2 = pathlib.Path(__file__).parent / "waldo_3.webp"



agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)
output = agent.run(input=(f"Please find Waldo in this image: {waldo_1}. "
                          "Waldo is a man with glasses wearing sweater with red and white stripes"))
print(output)
output = agent.run(input=(f"Great job! Now find Waldo in this image: {waldo_2}."))
print(output)
