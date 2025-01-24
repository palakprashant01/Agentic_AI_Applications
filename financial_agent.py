import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key_groq = os.getenv("GROQ_API_KEY")

# Check if the API key is loaded correctly
if not api_key_groq:
    raise ValueError("GROQ_API_KEY environment variable is not set")

# Create web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile", api_key=api_key_groq),  # Pass the API key
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Create financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile", api_key=api_key_groq),  # Pass the API key
    role="Get finance data",
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Combine both agents to create a multi AI agent
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id="llama-3.3-70b-versatile", api_key=api_key_groq),  # Pass the API key
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Test the multi AI agent with a simplified prompt
try:
    multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)
except Exception as e:
    print(f"An error occurred: {e}")