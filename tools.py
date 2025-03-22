from smolagents import CodeAgent, HfApiModel, tool, Tool

# Let's pretend we have a function that fetches the highest-rated catering services.
# @tool
# def catering_service_tool(query: str) -> str:
#     """
#     This tool returns the highest-rated catering service in Gotham City.
    
#     Args:
#         query: A search term for finding catering services.
#     """
    
#     services = {
#         "Gotham Catering Co.": 4.9,
#         "Wayne Manor Catering": 4.8,
#         "Gotham City Events": 4.7,
#     }
    
#     best_service = max(services, key=services.get)
    
#     return best_service


# agent = CodeAgent(tools=[catering_service_tool], model=HfApiModel())


# result = agent.run(
#     "Can you give me the name of the highest-rated catering service in Gotham City?"
# )

# print(result)   


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
  
  def forward(self, category:str):
    themes = {  "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }

    return themes.get(category.lower(), "No theme found for the specified category.")
  
party_theme_tool = SuperheroPartyThemeTool()
agent = CodeAgent(tools=[party_theme_tool], model=HfApiModel()) 
  
result = agent.run(
    "What would be a good superhero party idea for a 'villain masquerade' theme?"
)
print(result)