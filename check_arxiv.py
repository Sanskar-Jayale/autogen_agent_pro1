from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
import arxiv
from typing import List, Dict, AsyncGenerator
load_dotenv()

openai_brain = OpenAIChatCompletionClient(model = "gpt-4o", api_key=os.getenv('OPENAI_API_KEY'))


def arxiv_search(query:str, max_results:int = 5) -> List[Dict]:
    """Return a compact list of the arxiv paper matching *query*.
    Each element contains: ``title``, ``authors``, ``published``, ``summary`` and 
    ``pdf_url``.The helper is wrapped as an Autogen *FunctionTool* below so it 
    can be invoked by agent through the normal tool-use machanism."""
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    
    paper: List[Dict] = []
    for result in client.results(search):
        paper.append(
            {
                "title":result.title,
                "authors": [a.name for a in result.authors],
                "published":result.published.strftime("%Y-%m-%d"),
                "summary":result.summary,
                "pdf_url":result.pdf_url,
            }
        )
    return paper

print(arxiv_search(query="agents"))