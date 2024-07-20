# Sean Schumacher
# CRAG model
# July 17 2024

# A Model using CRAG approach built out in paper
# Allows for self reflection and self grading

import os
from dotenv import load_dotenv
import pprint
import streamlit as st

# Load the .env file
load_dotenv()

# env variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

from graph import app # import langgraph model

# Add LangSmith tracing here if needed
# Add later, lower priority

def main():
    st.title("CFI Application")

    # User input
    user_question = st.text_input("Enter your question:")

    if st.button("Generate Response"):
        if user_question:
            # Create a placeholder for the output
            output_placeholder = st.empty()

            # Initialize the output
            full_output = ""

            # Run the model
            inputs = {"question": user_question}
            for output in app.stream(inputs):
                for key, value in output.items():
                    # Node output
                    node_output = f"Node '{key}':\n"

                    # Optional: print full state at each node
                    # node_output += pprint.pformat(value["keys"], indent=2, width=80, depth=None)

                    node_output += "\n---\n"

            # Access final generation key
            if "generation" in value:
                full_output += pprint.pformat(value["generation"])
            else:
                full_output += "No 'generation' key found in output"

            # Final output display
            st.subheader("final Output:")
            st.text(full_output)
        else:
            st.warning("Please enter a question")

if __name__ == "__main__":
    main()