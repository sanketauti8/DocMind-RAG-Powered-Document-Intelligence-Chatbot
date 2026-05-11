"""
RAG Chatbot v2 — with Memory + Web Search Fallback

Three layers of knowledge:
  1. Your documents (ChromaDB) — searched first
  2. Web search (Tavily) — fallback when docs don't have the answer
  3. Conversation memory — for questions about the chat itself

SETUP:
  pip3 install ollama chromadb sentence-transformers tavily-python
  Make sure you've run step1_index_docs.py first to index your documents.
"""

import ollama
import chromadb
import json
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient


# ================================================================
# SETUP — load models and connect to databases
# ================================================================

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to the ChromaDB we created in step1
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("my_documents")
print(f"Connected to ChromaDB ({collection.count()} chunks indexed)")

# Tavily for web search — replace with your API key
tavily = TavilyClient(api_key="API_KEY")


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def retrieve(query, n_results=3):
    """
    Search ChromaDB for chunks relevant to the query.

    Returns:
      chunks: list of text chunks
      distances: list of distance scores (lower = more relevant)
    """
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )
    chunks = results["documents"][0]
    distances = results["distances"][0]
    return chunks, distances


def web_search(query):
    """
    Search the web using Tavily API.
    Returns a list of results with title, snippet, and url.
    """
    results = tavily.search(query=query, max_results=3)
    clean = []
    for r in results["results"]:
        clean.append({
            "title": r.get("title", ""),
            "snippet": r.get("content", "")[:500],
            "url": r.get("url", ""),
        })
    return clean


# ================================================================
# MAIN ASK FUNCTION — the brain of the chatbot
# ================================================================

def ask(question, conversation_history):
    """
    The RAG pipeline with three knowledge sources:

    1. Check if this is a question about our conversation → use memory
    2. Search documents → if relevant, use them
    3. Fall back to web search → if documents don't have the answer

    Args:
      question: what the user asked
      conversation_history: list of all previous messages

    Returns:
      answer: the model's response
      source: which knowledge source was used ("memory", "documents", or "web")
    """

    # ---- Step 0: Is this about our conversation? ----
    # These keywords suggest the user is asking about
    # something we discussed, not searching for new info.
    memory_keywords = [
        "last question", "previous question", "earlier",
        "we discussed", "you said", "i asked", "i said",
        "remember", "my name", "conversation", "we talked",
        "you told", "before", "you mentioned",
    ]
    is_memory_question = any(kw in question.lower() for kw in memory_keywords)

    if is_memory_question and conversation_history:
        # No search needed — the answer is in the conversation
        context = "This is a question about the conversation history. Use the conversation history to answer."
        source = "memory"
        print(f"\n  💬 This is about our conversation — using memory")

    else:
        # ---- Step 1: Search your documents ----
        print(f"\n  📎 Searching your documents...")
        chunks, distances = retrieve(question)
        best_distance = distances[0]
        print(f"  📎 Best match distance: {best_distance:.4f}")

        # ---- Step 2: Are the documents relevant enough? ----
        # Distance < 0.8 = documents probably have the answer
        # Distance >= 0.8 = documents are too far off, try the web
        if best_distance < 0.8:
            # DOCUMENTS HAVE THE ANSWER
            context = "\n\n---\n\n".join(chunks)
            source = "documents"
            print(f"  ✓  Found relevant chunks in your documents")
        else:
            # FALL BACK TO WEB SEARCH
            print(f"  ✗  Documents don't seem relevant (distance {best_distance:.2f})")
            print(f"  🌐 Searching the web instead...")
            web_results = web_search(question)
            context = "\n\n---\n\n".join(
                f"Title: {r['title']}\n{r['snippet']}\nSource: {r['url']}"
                for r in web_results
            )
            source = "web"
            print(f"  🌐 Got {len(web_results)} web results")

    # ---- Step 3: Pick the right system prompt ----
    # The system prompt changes based on where the answer came from.
    # This helps the model understand what kind of context it's seeing.
    if source == "memory":
        system_msg = (
            "You are a helpful assistant. The user is asking about your conversation. "
            "Answer based on the conversation history. Be accurate about what was said."
        )
    elif source == "documents":
        system_msg = (
            "You are a helpful assistant that answers questions based on the provided documents. "
            "Answer based on the CONTEXT below. If the context doesn't contain the answer, say so. "
            "You also remember the full conversation history."
        )
    else:
        system_msg = (
            "You are a helpful assistant. The user's documents didn't have the answer, "
            "so you're using web search results instead. Answer based on the WEB RESULTS below. "
            "Mention that this came from the web, not from their documents. "
            "You also remember the full conversation history."
        )

    # ---- Step 4: Build messages with full conversation history ----
    # The model sees: system prompt → all previous turns → current question
    # This is how it "remembers" the conversation.
    messages = [{"role": "system", "content": system_msg}]

    # Add all previous conversation turns
    for turn in conversation_history:
        messages.append(turn)

    # Add the current question with context
    user_msg = f"""CONTEXT ({source}):
{context}

QUESTION: {question}"""

    messages.append({"role": "user", "content": user_msg})

    # ---- Step 5: Get answer from Llama ----
    response = ollama.chat(
        model="llama3.1",
        messages=messages,
    )

    answer = response.message.content

    # ---- Step 6: Save this turn to conversation history ----
    # We save the simple question (not the context-stuffed version)
    # so the history stays clean and doesn't eat up context window.
    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    return answer, source


# ================================================================
# INTERACTIVE CHAT LOOP
# ================================================================

def main():
    print()
    print("=" * 60)
    print("  RAG CHATBOT v2")
    print("  + Conversation memory")
    print("  + Web search fallback")
    print("=" * 60)
    print()
    print("  📎 Ask about your documents — it searches them first")
    print("  🌐 If docs don't have the answer, it searches the web")
    print("  💬 It remembers your whole conversation")
    print()
    print("  Type 'quit' to exit")
    print("  Type 'clear' to reset conversation memory")
    print()

    # This list stores every message in the conversation.
    # It grows with each turn and gets sent to the model every time,
    # so the model can "remember" what was said before.
    conversation_history = []

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "bye"):
            print("\nGoodbye!")
            break
        if question.lower() == "clear":
            conversation_history = []
            print("  Memory cleared!\n")
            continue

        answer, source = ask(question, conversation_history)

        # Show which source was used
        icons = {"documents": "📎", "web": "🌐", "memory": "💬"}
        icon = icons.get(source, "📎")
        print(f"\nAssistant ({icon}): {answer}\n")


if __name__ == "__main__":
    main()