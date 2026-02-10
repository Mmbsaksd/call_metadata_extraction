from typing import Literal
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
 
load_dotenv()

class CallSummary(BaseModel):
    employee_name: str
    employee_id: str
    agent_name: str
    summary: str
    sentiment: Literal["good", "bad", "normal"]

INSTRUCTIONS = """You are an information extraction assistant. Extract structured metadata from a service desk call transcript into the provided schema.

SCHEMA FIELDS (fill exactly as specified):
- employee_name: Full name of the caller/employee (e.g., "First Last"). If not present, return "".
- employee_id: The numeric employee identifier explicitly stated or confirmed during authentication. Digits only; strip spaces and punctuation. If not present, return "".
- agent_name: The primary service desk agent’s name (the person who greeted and handled the call). If missing or unclear, return "".
- summary: A concise 2–4 sentence, factual summary of the issue and outcome in neutral tone. Include the user’s goal, key systems/apps mentioned, steps guided by the agent, and the final result (resolved/unresolved/next steps).
- sentiment: One of {{"good","bad","normal"}} reflecting the caller’s overall sentiment at the end:
  - "good": resolved and/or caller expresses thanks/satisfaction.
  - "bad": unresolved issue, escalation/abandonment, or clear frustration.
  - "normal": neutral/mixed, outcome unclear, or insufficient evidence.

DECISION & NORMALIZATION RULES:
1) Names: Capitalize first and last names (e.g., "jose" → "Jose"). If multiple agents appear, select the one who greets and completes most of the interaction.
2) Employee ID: Prefer the numeric ID explicitly provided by the caller or verified by the agent. If an email-like string appears, do NOT use that—only the numeric ID. Keep digits only (e.g., "93 45-4472" → "93454472").
3) Evidence Priority: Prefer explicit statements from the transcript over assumptions. If a field is not explicitly present, set it to "" (except sentiment, which must be "good", "bad", or "normal").
4) Summary Content: Be specific about the system(s) (e.g., Workday, SSO portal) and the key action(s) (e.g., password reset, username format, URL used, MFA). Avoid small talk and filler. Mention resolution status.
5) Conflict Handling: If conflicting info exists (e.g., two names/IDs), prefer the most recently confirmed item by the agent and caller.
6) Safety: Do not introduce information not present in the transcript. Do not guess phone numbers, emails, or departments unless explicitly stated.
7) Noise & ASR Artifacts: Ignore stutters, repetitions, and filler; extract the clean underlying facts.

OUTPUT:
Return only the structured object matching the schema. The calling framework will handle JSON formatting.

TRANSCRIPT:
{transcript}
"""

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0,
    max_tokens=500,
    timeout=60,
)

path = r"C:\Users\mohammed.a\Desktop\call_metadata_extraction\transcripts 1.csv"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

prompt = ChatPromptTemplate.from_messages([("system", INSTRUCTIONS)])
structured_llm = llm.with_structured_output(CallSummary)
chain = prompt | structured_llm

result: CallSummary = chain.invoke({"transcript": text})
print(result.model_dump())  # JSON-like dict