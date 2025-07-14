#agent most important tool in AI. Agent is context of Langchain. Deep Dive and core concepts
#agent is a tool that can be used to perform tasks, make decisions, and interact with the environment.
#Agents can be used to automate tasks, make decisions, and interact with the environment.

#packages to install: pip install -qU langchain-core==0.3.33 langchain-openai==0.3.3 langchain-community==0.3.16 langsmith==0.3.4 google-search-results==2.4.2
import os
from getpass import getpass

api_key = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "my_project"