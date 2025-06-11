from dotenv import load_dotenv
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel 
from agents.run import RunConfig
import asyncio

# Load environment variables
load_dotenv()

async def main():
    # Get secret key
    secret_key = os.getenv("GEMINI_API_KEY")

    # Model and client setup
    MODEL_NAME = "gemini-2.0-flash"

    external_client = AsyncOpenAI(
        api_key=secret_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    
    model = OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=external_client
    )

    config=RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True)

    # Create the agent
    assistance = Agent(
        name="Assistance",
        instructions="You are Assistance",  # fixed typo from 'instrucations'
        model=model
    )

    # Run the agent
    result = await Runner.run(
        starting_agent=assistance,
        input="Hello, how are you?",
        run_config=config
    )

    print(result.final_output)

if __name__ == '__main__':
   asyncio.run(main())
