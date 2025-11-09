# --- Imports ---
import os # For environment variables
import time
import argparse
import sys
from pathlib import Path
import json
import warnings
import re
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch # Import torch for tensor operations

# --- Langchain/OpenAI Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage,SystemMessage, HumanMessage # To handle output type

def get_llm_context(pert,gene,cell_type):

    llm = ChatOpenAI(
        model="openai_o3_mini",
        api_key="xxxxx",
        base_url="https://api.marketplace.novo-genai.com/v1",
        max_retries=3,
        request_timeout=120,
    )

    system_prompt_text = (
            "You are a knowledgeable bioinformatics expert specializing in gene regulation and pathway analysis. "
            "Your task is to retrieve and synthesize information about the functional relationship between genes, "
            "particularly in the context of perturbations and specific cell types. Focus on known pathways, biological processes, "
            "and molecular functions relevant to the user's query. Provide a concise summary suitable as background context "
            "for analyzing regulatory effects."
        )

    user_query_text = (
        f"Summarize the key functional annotations, pathways (e.g., KEGG, GO terms, Reactome), and biological processes "
        f"associated with the interplay between the gene '{pert}' (specifically when perturbed, like knockdown/knockout) "
        f"and the target gene '{gene}' within the context of '{cell_type}' cells. "
        f"Highlight information relevant to understanding potential regulatory effects of perturbing '{pert}' on '{gene}'."
    )
    # --- Construct the list of messages ---
    messages = [
        SystemMessage(content=system_prompt_text),
        HumanMessage(content=user_query_text),
    ]

    response = llm.invoke(messages)

    if isinstance(response, AIMessage):
        return "ChatGPT o3 mini", response.content
    else:
        logging.error(f"Unexpected response type from LLM: {type(response)}")
