#agent most important tool in AI. Agent is context of Langchain. Deep Dive and core concepts
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
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.com"
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
    ("system", "You're a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

#Next we need to define our LLM, we will use the gpt-4o-mini model with the temprature of 0.0

from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = api_key