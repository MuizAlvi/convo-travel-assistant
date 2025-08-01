
# ✈️ AI Travel Assistant

A LangChain-based conversational agent that helps users plan international travel by:

- 🔍 Finding and filtering flights based on user preferences
- 🛂 Answering visa requirement questions using Retrieval-Augmented Generation (RAG)
- 💵 Providing refund and cancellation policy details

---

## 📁 System Overview

### Key Features
- **Natural Language Understanding**: Parses vague queries like _"Find me a Star Alliance flight to Tokyo in August"_.
- **LLM-Powered Search**: Uses GPT to extract structured search criteria from free-form queries.
- **Agentic Architecture**: Routes user requests to tools using a LangChain agent.
- **RAG for Policy Questions**: Answers visa and refund queries using a vector database (ChromaDB).
- **Gradio UI**: Clean browser-based chat interface.

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-travel-assistant.git
cd ai-travel-assistant
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env` file

```env
OPENAI_API_KEY=your_openai_key_here
```

### 4. Run the app

```bash
python gradio_app.py
```

App runs at: [http://localhost:7860](http://localhost:7860)

---

## 🧠 Prompt Strategy

We use a **two-level prompt flow**:

### 🔸 Flight Search

An LLM parses the user query to extract structured filters like:

```json
{
  "from": "Dubai",
  "to": "Tokyo",
  "departure_month": "August",
  "alliance": "Star Alliance",
  "overnight_layover": false
}
```

These are applied against a JSON-based mock flight listing.

### 🔸 RAG (Visa & Refund)

We use `langchain.chains.RetrievalQA` with a Chroma vectorstore and OpenAI embeddings to answer:

* "Can UAE citizens travel to Japan without a visa?"
* "What’s the refund policy on refundable tickets?"

---

## 🧩 Agent Logic

* Agent Type: `zero-shot-react-description`
* Tools Registered:

  * `FlightSearch`: filters flights from mock JSON
  * `VisaRAG`: answers visa queries via vector search
  * `PolicyRAG`: answers refund policy questions

The agent selects a tool based on the query type, runs it, and returns the result.

---

## 🧪 Sample Outputs

### ✈️ Flight Search

**User:** Find me a round-trip to Tokyo in August with Star Alliance airlines only. I want to avoid overnight layovers.
**Response:**

> The best option is a Turkish Airlines flight from Dubai to Tokyo departing on August 15th and returning on August 30th for \$950.

---

### 🛂 Visa Question

**User:** Can UAE citizens travel to Japan without a visa?
**Response:**

> UAE citizens can travel to Japan without a visa for up to 30 days for tourism purposes.

---

### 💵 Refund Policy

**User:** Can I get a refund on my ticket?
**Response:**

> Refundable tickets can be canceled up to 48 hours before departure, subject to a 10% processing fee.

---

## 🔧 Folder Structure

```
├── data/
│   ├── flights.json
│   └── visa_rules.md
├── main.py              # Agent, tools, RAG setup
├── gradio_app.py        # Gradio interface
├── requirements.txt
└── README.md
```

---

## 🧑‍💻 Author

Built by Muiz Alvi as a demonstration of LLM-based travel planning using RAG and Agentic AI