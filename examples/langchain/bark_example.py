import os

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY must be set")

from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from gradio_tools.tools import BarkTextToSpeechTool, StableDiffusionTool, StableDiffusionPromptGeneratorTool

from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
tools = [BarkTextToSpeechTool().langchain,
         StableDiffusionTool().langchain,
         StableDiffusionPromptGeneratorTool().langchain]


agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)
output = agent.run(input=("Please create a jingle for a spanish company called 'Chipi Chups' that makes lollipops. "
                          "The jingle should be catchy and playful and meant to appeal to all ages."))
print(output)
output = agent.run(input=("Now create a logo for this company. Please improve"))
print(output)
