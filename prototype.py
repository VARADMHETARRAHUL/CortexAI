import os
import json
import logging
import re
from typing import Any, Dict, List
from collections import defaultdict

import httpx
from mcp.server.fastmcp import FastMCP

prompt_system = """
You are an expert graph builder. Given a text prompt, extract key concepts and relationships.
"""

mcp = FastMCP()

# ---- Simple heuristic graph builder for now ----

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "is", "are", "was", "were", "be", "being", "been",
    "this", "that", "these", "those", "as", "it", "its", "into",
    "about", "over", "under", "up", "down", "out", "so", "such",
}

def build_graph_from_text(text: str) -> Dict[str, Any]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    concepts: List[str] = [t for t in tokens if t not in STOPWORDS]

    unique_concepts = sorted(set(concepts))
    node_id_map = {concept: i for i, concept in enumerate(unique_concepts)}

    nodes = [
        {
            "id": node_id_map[c],
            "label": c,
            "type": "concept",
        }
        for c in unique_concepts
    ]

    edge_counts = defaultdict(int)
    for i in range(len(concepts) - 1):
        src = concepts[i]
        dst = concepts[i + 1]
        if src == dst:
            continue
        a, b = sorted([src, dst])
        edge_counts[(a, b)] += 1

    edges = []
    for (src, dst), count in edge_counts.items():
        edges.append(
            {
                "source": node_id_map[src],
                "target": node_id_map[dst],
                "weight": count,
                "relation": "co_occurs",
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "original_text_length": len(text),
            "num_tokens": len(tokens),
            "num_concepts": len(unique_concepts),
        },
    }


# ---- Free LLM Agent (async) using OpenRouter or DeepSeek ----

LLM_API_URL = "https://api.deepseek.com/chat/completions"
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_KEY_HERE")


async def llm_agent(prompt: str) -> str:
    """
    Calls a free/cheap LLM endpoint asynchronously and returns its text response.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(LLM_API_URL, headers=headers, json=payload)
        data = resp.json()

        try:
            return data["choices"][0]["message"]["content"]
        except:
            return "LLM error or invalid response."


# ---- MCP Tool: Graph Builder ----

@mcp.tool()
def transformer_agent(text: str) -> Dict[str, Any]:
    graph = build_graph_from_text(text)
    return graph


# ---- MCP Tool: Converts prompt ➜ Graph + LLM reasoning ----

@mcp.tool()
async def prompt_to_agent(prompt: str) -> dict:
    """
    1. Builds graph from the user prompt
    2. Calls async LLM agent to extract structured reasoning
    3. Returns both
    """

    graph = build_graph_from_text(prompt)

    llm_response = await llm_agent(prompt)

    return {
        "message": "Graph + LLM reasoning generated.",
        "graph": graph,
        "llm_reasoning": llm_response
    }
@mcp.tool()
async def push_llm_engine(prompt:str)-> str:
    """
    Pushes the LLM through multiple states (nodes) to and fro while thinking 
    """
    result = await prompt_to_agent(prompt)
    graph = result["graph"]
    
    states = []

    for node in graph['nodes']:
        concept = node['label']
        
        reasoning = await llm_agent(
            f'You are moving through concept like a reasoning agent.'
            f'Current concept:{concept}\n'
            f'Explain how this concept connects to broader meaning of {prompt}'
        )

        states.append({
            'node':concept,
            'reasoning':reasoning
        })
    return {
        'message':'LLM was moved forward and backward through concept nodes.',
        'graph':graph,
        'states':states
    } 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
