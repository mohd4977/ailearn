#agent most important tool in AI. Agent is context of Langchain. Deep Dive and core concepts
#Youtube: https://www.youtube.com/watch?v=Cyv-dgv80kE&t=5040s
#agent is a tool that can be used to perform tasks, make decisions, and interact with the environment.
#Agents can be used to automate tasks, make decisions, and interact with the environment.

#packages to install: pip install -qU langchain-core==0.3.33 langchain-openai==0.3.3 langchain-community==0.3.16 langsmith==0.3.4 google-search-results==2.4.2
import os
from dotenv import load_dotenv
from getpass import getpass
import json

# Load the .env file from the current directory
load_dotenv()

# Get the API key from .env or ask user if not found
api_key = os.getenv("LANGCHAIN_API_KEY")

if not api_key:
    raise ValueError("API key is required!")

# Set other LangChain environment variables
os.environ["LANGCHAIN_API_KEY"] = api_key
os.environ["LANGCHAIN_TRACING_V2"] = "false" #set to true if you want to trace the agent execution
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com" #activate if want tracing on langsmith dashboard. It is very expensive.
os.environ["LANGCHAIN_PROJECT"] = "my_project"

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

"""
ChatPromptTemplate
Used to define structured prompts for chat models.

Supports message roles (system/user/assistant).

You can build prompts like:

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful assistant."),
    ("user", "{input}")
])

MessagesPlaceholder
A special placeholder used for inserting past chat messages (for memory).

Often used like:

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

"""

@tool #with this decorator our function becomes Structured tool object
def add(x: float, y: float, z: float | None = 0.3) -> float: #just like this there are more tools multiply, exponentiate, subtract
    """Add 'x' and 'y'."""
    return x + y if z else x + y + z

@tool #with this decorator our function becomes Structured tool object
def subtract(x: float, y: float, z: float | None = 0.3) -> float: #just like this there are more tools multiply, exponentiate, subtract
    """Subtract 'y' from 'x'."""
    return x - y if z else x - y - z

print(add) #structure tool will output something like name='add' description="Add 'x' and 'y'." args_schema=<class 'langchain_core.utils.pydantic.add'> func=<function add at 0x000001E9A6D654E0>
#you can print name and description of function

print(add.name)
print(add.description)

#to print schema you can use args schema model json schema
print(add.args_schema.model_json_schema())

llm_output_string = "{\"x\":5,\"y\":2}" #this is the output from the LLM
llm_output_dict = json.loads(llm_output_string) #change the output to dict
print(llm_output_dict)

print(add.func(**llm_output_dict))

#defining prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("user", "{input}"),
    ("user", "{agent_scratchpad}")  # required!
])

#Next we need to define our LLM, we will use the gpt-4o-mini model with the temprature of 0.0

from langchain_openai import ChatOpenAI

ai_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = ai_key

llm = ChatOpenAI(model="gpt-3.5-turbo")  # or "gpt-4o"
#response = llm.invoke("What's something cool about LangChain?")
#print(response.content)

#We'll be using the older ConversationBufferMemory class to store the chat history rather than the newer RunableWithMessageHistory class.

from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.agents import create_tool_calling_agent

# Now we will initialize the agent with the tools, prompt, LLM, and memory.
tools = [add, subtract] #list of tools to be used by the agent

memory = ConversationBufferMemory(
    memory_key="chat_history", #Must align with the MessagesPlaceholder variable_name
    return_messages=True #To return the messages in the chat history
)

agent = create_tool_calling_agent(
    tools=tools,
    prompt=prompt,
    llm=llm,
)

agent_with_memory = RunnableWithMessageHistory(
    agent,
    lambda session_id: InMemoryChatMessageHistory(),
    input_messages_key="input",
    history_messages_key="agent_scratchpad"
)










#Our agent by itself is like one-step of our agent execution loop. So, if we call the agent.invoke() method, 
# it will execute the agent with the current input and return the output. no tools will be executed yet,
#because we haven't provided any input to the agent yet and no next iteration of the agent has been executed.

response = agent.invoke({ #Old way to invoke the agent
    "input": "What is 5 + 2?",
    "intermediate_steps": []
})
print(response)

response2 = agent_with_memory.invoke(
    {
        "input": "What is 5 + 2?",
        "intermediate_steps": []  # required
    },
    config={"configurable": {"session_id": "user-session-1"}}
)

print(response2)

#Why Use AgentExecutor?
#Simplifies how you run agents.

#Automatically handles the intermediate_steps lifecycle.

#Integrates easily with memory (ConversationBufferMemory).

#Supports multi-step execution out of the box.

from langchain.agents import AgentExecutor

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# Invoke like this
response3 = agent_executor.invoke({
    "input": "What is 5 - 2?"
})

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(response3)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

#from langsmith import Client #To test if Lanchain is connected to LangSmith
# Uncomment the following lines to connect to LangSmith
#client = Client()
#print("LangSmith connected:", list(client.list_projects()))