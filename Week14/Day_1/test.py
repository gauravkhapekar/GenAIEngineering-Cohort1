from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENROUTER_API_KEY")

# Check if API key exists
if not openai_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

# ===== 2. Initialize LLM via OpenRouter =====
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=openai_api_key
)

# ===== 3. PROMPT TEMPLATES =====
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize this text in one sentence:\n{text}"
)

keywords_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Extract 3 key keywords from this summary:\n{summary}"
)

# ===== 4. SIMPLE LLM CHAIN =====
summary_chain = LLMChain(
    llm=llm,
    prompt=summary_prompt,
    output_key="summary"
)

# ===== 5. SEQUENTIAL CHAIN =====
keywords_chain = LLMChain(
    llm=llm,
    prompt=keywords_prompt,
    output_key="keywords"
)

overall_chain = SequentialChain(
    chains=[summary_chain, keywords_chain],
    input_variables=["text"],
    output_variables=["summary", "keywords"]
)

# ===== 6. RUN THE CHAIN =====
input_text = "LangChain is a framework that helps developers build AI-powered apps by connecting LLMs to data, memory, and external tools."

try:
    result = overall_chain.invoke({"text": input_text})
except Exception as e:
    # Fallback to run() for older versions
    result = overall_chain.run({"text": input_text})

print("\n--- SUMMARY ---")
print(result["summary"])
print("\n--- KEYWORDS ---")
print(result["keywords"])