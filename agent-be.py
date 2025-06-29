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

    
    
    
# arxiv_researcher_agent = AssistantAgent(
#     name = "arxiv_researcher_agent",
#     description="the agent create arXivthe queies and retrives candidates papers",
#     model_client=openai_brain,
#     tools= [arxiv_search],
#     system_message=''
    
# )

# summarizer_agent = AssistantAgent(
#     name = 'summarizer-agent',
#     description="the agent ahich summarizer the result",
#     model_client=openai_brain,
#     system_message=(
#         "You are an expert researcher. when you recive the JSON list of"
#         "papers, with a literature-review style report in Markdown:\n"\
#         "1. start with 2-3 sentences of introduction of the topic.\n "\
#         "2. Then include one bullets per paper with:title (as Markdown"
#         "link), authors, the specific problem problem trackled, and its key"
#         "contribution.\n"\
#         "3. Close with a single-sentence takeaway."
#     ),
    
# )

print(arxiv_search(query="agentic_ai"))