from dotenv import load_dotenv
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

# Load environment variables
load_dotenv()

def main():
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

    # Create the agent
    assistance = Agent(
        name="Assistance",
        instructions="You are Assistance",  # fixed typo from 'instrucations'
        model=model
    )

    # Run the agent
    result = Runner.run_sync(
        starting_agent=assistance,
        input="Hello, how are you?"
    )

    print(result.final_output)

if __name__ == '__main__':
    main()
