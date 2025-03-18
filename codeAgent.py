from huggingface_hub import login
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, tool
import numpy as np
import datetime
import time
from dotenv import load_dotenv
import base64
import os

##################################################################################

# code to track the progress and logs of llm calls, using lang fuse and open telemetry

from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

login()
load_dotenv()

LANGFUSE_AUTH=base64.b64encode(f"{os.environ["LANGFUSE_PUBLIC_KEY"]}:{os.environ["LANGFUSE_SECRET_KEY"]}".encode()).decode()
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] =  os.environ["ENDPOINT"]
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"


trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

################################################################

# agent = CodeAgent(tools=[DuckDuckGoSearchTool()],model=HfApiModel())

# agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")


# the ai agent is capable of using locally made functions as tools too
# @tool
# def suggest_menu(occasion:str) -> str:
#     """
#     Suggests a menu based on the occasion.
#     Args:
#         occasion (str): The type of occasion for the party. Allowed values are:
#                         - "casual": Menu for casual party.
#                         - "formal": Menu for formal party.
#                         - "superhero": Menu for superhero party.
#                         - "custom": Custom menu.
#     """
#     if occasion == "casual":
#         return "Pizza, snacks, and drinks."
#     elif occasion == "formal":
#         return "3-course dinner with wine and dessert."
#     elif occasion == "superhero":
#         return "Buffet with high-energy and healthy food."
#     else:
#         return "Custom menu for the butler."

# # Alfred, the butler, preparing the menu for the party
# agent = CodeAgent(tools=[suggest_menu], model=HfApiModel())

# # Preparing the menu for the party
# agent.run("Prepare a formal menu for the party.")
      
agent = CodeAgent(tools=[],model=HfApiModel(),additional_authorized_imports=['datetime'])

agent.run(
    """
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 45 minutes

    If we start right now, at what time will the party be ready?
    """
)

# example of code agent using smolAgents

from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, HfApiModel, Tool, tool, VisitWebpageTool

@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion: The type of occasion for the party.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.
    
    Args:
        query: A search term for finding catering services.
    """
    # Example list of catering services and their ratings
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }
    
    # Find the highest rated catering service (simulating search query filtering)
    best_service = max(services, key=services.get)
    
    return best_service

class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""
    
    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
        }
    }
    
    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }
        
        return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")


# Alfred, the butler, preparing the menu for the party
agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(), 
        VisitWebpageTool(),
        suggest_menu,
        catering_service_tool,
        SuperheroPartyThemeTool(),
        FinalAnswerTool()
    ], 
    model=HfApiModel(),
    max_steps=10,
    verbosity_level=2
)

agent.run("Give me the best playlist for a party at the Wayne's mansion. The party idea is a 'villain masquerade' theme")