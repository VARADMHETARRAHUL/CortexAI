from dotenv import load_dotenv
load_dotenv()

import os
import httpx
from mcp.server.fastmcp import FastMCP
import asyncio
import json

API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"

# Required OpenRouter headers
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "http://localhost",
    "X-Title": "LLMspeaks",
    "Content-Type": "application/json"
}

mcp = FastMCP()

# -------------------------------
# TOOL 1 — Grok
# -------------------------------
@mcp.tool()
async def call_grok(prompt: str) -> str:
    """
    Call Grok model via OpenRouter.
    """
    data = {
        "model": "x-ai/grok-4.1-fast:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(URL, headers=HEADERS, json=data)
        return json.dumps(r.json())

# -------------------------------
# TOOL 2 — Mistral
# -------------------------------
@mcp.tool()
async def call_mistral(prompt: str) -> str:
    """
    Call Mistral model via OpenRouter.
    """
    data = {
        "model": "mistralai/mistral-small-3.1-24b-instruct:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(URL, headers=HEADERS, json=data)
        return json.dumps(r.json())

# -------------------------------
# TOOL 3 — Debate Engine
# -------------------------------
@mcp.tool()
async def debate_llm(prompt: str) -> str:
    """
    Debate between Grok and Mistral and produce a final synthesized answer.

    Args:
        prompt (str): The topic or question to debate.
    Returns:
        str: Debate transcript and final conclusion.
    """
    rounds = 3
    grok_claim = prompt
    mistral_claim = prompt

    grok_text = ""
    mistral_text = ""

    for _ in range(rounds):

        grok_resp_str, mistral_resp_str = await asyncio.gather(
            call_grok(grok_claim),
            call_mistral(mistral_claim)
        )

        grok_resp = json.loads(grok_resp_str)
        mistral_resp = json.loads(mistral_resp_str)

        # If request failed, return error details instead of crashing
        if "choices" not in grok_resp:
            return f"Grok API error:\n{grok_resp}"

        if "choices" not in mistral_resp:
            return f"Mistral API error:\n{mistral_resp}"

        grok_text = grok_resp["choices"][0]["message"]["content"]
        mistral_text = mistral_resp["choices"][0]["message"]["content"]

        grok_claim = f"Mistral said: {mistral_text}\nRespond with your counter-argument:"
        mistral_claim = f"Grok said: {grok_text}\nRespond with your counter-argument:"

    # Final synthesis by Grok
    final_judge_str = await call_grok(
        f"""Given the final debate, combine the strongest points from both models.

Grok Final: {grok_text}
Mistral Final: {mistral_text}

Provide one unified final answer.
"""
    )

    final_judge = json.loads(final_judge_str)

    if "choices" not in final_judge:
        return f"Final Judge API Error:\n{final_judge}"

    final_text = final_judge["choices"][0]["message"]["content"]

    return f"""
=== LLM Debate Complete ===

Grok Final Answer:
{grok_text}

Mistral Final Answer:
{mistral_text}

=== Synthesized Final Answer ===
{final_text}
"""

# -------------------------------
# Start MCP Server
# -------------------------------
def main():
    if not API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY environment variable!")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()