import spacy
from transformers import pipeline

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load a pre-trained model for answering questions
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Define a knowledge base for context
knowledge_base = {
    "python": "Python is a high-level programming language known for its readability and versatility.",
    "nlp": "Natural Language Processing (NLP) enables machines to understand and respond to human language.",
    "chatbot": "A chatbot is an AI-powered tool designed to simulate human conversation."
}

# Function to extract keywords and answer questions
def get_answer(user_query):
    doc = nlp(user_query.lower())
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]

    for keyword in keywords:
        if keyword in knowledge_base:
            return knowledge_base[keyword]
    
    # If no keyword matches, use a fallback model for open-ended questions
    context = " ".join(knowledge_base.values())
    result = qa_pipeline(question=user_query, context=context)
    return result['answer'] if result['score'] > 0.5 else "Sorry, I don't have an answer for that."

# Chat loop
def chatbot():
    print("Chatbot is ready! Type 'exit' to end the chat.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        response = get_answer(user_query)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chatbot()
