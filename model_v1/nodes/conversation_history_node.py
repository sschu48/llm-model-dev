def summarize_conversation(history):
    # this is where you can create a method for summarizing chat history
    # this could get pretty detailed...
    # for now it keeps the last three exchanges
    return " ".join(history[-3:])

def manage_conversation_history(state):
    current_question = state["question"]
    current_generation = state["generation"]

    # retrieve existing history or initialize if it doe
    history = state.get("conversation_history", "").split(" [SEP] ") if state.get("conversation_history") else []

    # add current exchange to history
    if current_question or current_generation:
        history.append(f"Q: {current_question} A: {current_generation}")

    # update state with new summarized history
    state["conversation_history"] = summarize_conversation(history)

    return state