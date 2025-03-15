import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")

client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")


# simply prompting the llm

# output = client.text_generation(
#     "The capital of France is",
#     max_new_tokens=100,
# )
# print(output)


# If we now add the special tokens related to Llama3.2 model, the behaviour changes and is now the expected one.

# prompt="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

# The capital of france is<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# """
# output = client.text_generation(
#     prompt,
#     max_new_tokens=100,
# )

# print(output)


# but we cant remember these tokens, right? so we use templates of the hugging face library

# output = client.chat.completions.create(
#     messages=[
#         {"role": "user", "content": "The capital of france is"},
#     ],
#     stream=False,
#     max_tokens=1024,
# )

# print(output.choices[0].message.content)


# now that we know about how things are working as tokens, lets see how system prompt works, shall we?

# This system prompt is a bit more complex and actually contains the function description already appended.
# Here we suppose that the textual description of the tools has already been appended
SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use :
```
{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:
```
$JSON_BLOB
```
Observation: the result of the action. This Observation is unique, complete, and the source of truth.
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """

# Dummy function
def get_weather(location):
    return f"the weather in {location} is sunny with low temperatures. \n"


prompt=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}
<|eot_id|><|start_header_id|>user<|end_header_id|>
What's the weather in London ?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

output = client.text_generation(
  prompt=prompt,
  max_new_tokens=200,
  stop=["Observation:"]
)

new_prompt=prompt+output+get_weather('London')

# print(new_prompt)

final_output = client.text_generation(
  new_prompt,
  max_new_tokens=200
)

print(final_output)

