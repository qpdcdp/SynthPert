import csv
import pandas as pd
import numpy as np
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from datasets import load_dataset, DatasetDict, interleave_datasets
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Disable logging from the httpx module
logging.getLogger('httpx').setLevel(logging.WARNING)

# Set a random seed for reproducibility
np.random.seed(88)


# Initialize LLM
model_name = 'openai_o3_mini'
llm = ChatOpenAI(
    model=model_name,
    api_key="sk-7lN2yHKbX5NWkbzjFU0faQ",
    base_url="https://api.marketplace.novo-genai.com/v1",
)

# Define Pydantic models
class SQL_QA(BaseModel):
    thinking: Optional[str] = Field(default=None, description="Thinking for generating SQL")
    sql: Optional[str] = Field(default=None, description="The SQL used to answer the question")

class SQL_evaluator(BaseModel):
    grade: Optional[str] = Field(default=None, description="The SQL evaluation")

# Set up parsers
parser = PydanticOutputParser(pydantic_object=SQL_QA)
parser_eval = PydanticOutputParser(pydantic_object=SQL_evaluator)

dataset_scc_train = load_dataset("b-mc2/sql-create-context", split="train[:80%]")
dataset_scc_test = load_dataset("b-mc2/sql-create-context", split="train[-20%:-10%]")
dataset_scc_val = load_dataset("b-mc2/sql-create-context", split="train[-10%:]")

dataset_tts_train = load_dataset("Clinton/Text-to-sql-v1", split="train[:80%]")
dataset_tts_train = dataset_tts_train.remove_columns(["source", "text"])
dataset_tts_train = dataset_tts_train.rename_columns(
    {"instruction": "question", "input": "context", "response": "answer"}
)
dataset_tts_test = load_dataset("Clinton/Text-to-sql-v1", split="train[-20%:-10%]")
dataset_tts_test = dataset_tts_test.remove_columns(["source", "text"])
dataset_tts_test = dataset_tts_test.rename_columns(
    {"instruction": "question", "input": "context", "response": "answer"}
)
dataset_tts_val = load_dataset("Clinton/Text-to-sql-v1", split="train[-10%:]")
dataset_tts_val = dataset_tts_val.remove_columns(["source", "text"])
dataset_tts_val = dataset_tts_val.rename_columns(
    {"instruction": "question", "input": "context", "response": "answer"}
)

dataset_ks_train = load_dataset("knowrohit07/know_sql", split="validation[:80%]")
dataset_ks_test = load_dataset("knowrohit07/know_sql", split="validation[-20%:-10%]")
dataset_ks_val = load_dataset("knowrohit07/know_sql", split="validation[-10%:]")

dataset = DatasetDict(
    {
        "train": interleave_datasets(
            [dataset_scc_train, dataset_tts_train, dataset_ks_train]
        ),
        "test": interleave_datasets(
            [dataset_scc_test, dataset_tts_test, dataset_ks_test]
        ),
        "validation": interleave_datasets(
            [dataset_scc_val, dataset_tts_val, dataset_ks_val]
        ),
    }
)

train_dataset = dataset["train"]

def process_data_row(data):
    """Process a single data row for SQL QA and evaluation."""
    schema = data["context"]
    correct_sql = data["answer"]
    
    # Create QA prompt
    create_qa = [
        ("system",
                f"""You are a SQL expert, Below are sql tables schemas paired with instruction that describes a task
                using valid SQLite, write a response that appropriately completes the request for the provided tables.
                SCHEMA: {schema}. When answering provide a reasoing for the sql query such that you use following
                template:
                <thinking> /<thinking> 
                
                <sql> /<sql> 
        """,),
        (data["question"]),
    ]
    
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    ans = chain.invoke(create_qa)

    sql_eval = [
        ("system", f"""
                You are SQL expert and your task is to evaluate if if the sql querry{ans.sql} is correct 
                based on the Schema and the correct SQL
                Schema: {schema}
                correct: {correct_sql}

                Return ONLY "Correct" or "Wrong"
                    """),
        (ans.sql),
    ]
    
    eval_prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser_eval.get_format_instructions()},
    )
    
    eval_chain = eval_prompt | llm | parser_eval
    eval_res = eval_chain.invoke(sql_eval)

    return [
        data["answer"],
        data["question"],
        data["context"],
        ans.thinking,
        ans.sql,
        eval_res.grade,
    ]

# Write results to CSV
with open("sql_qa_multi_head_{ver}.csv", mode="a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    count=0
    # Write header row
    writer.writerow(["answer", "question", "context", "thinking", "Response", "Eval"])

    # Process rows in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:  # adjust number of workers as needed
        futures = {executor.submit(process_data_row, data): data for data in train_dataset}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
            try:
                result = future.result()
                writer.writerow(result)
                file.flush()
                count+=1
                print(count)
            except Exception as e:
                print(f"Error processing row: {str(e)}")

if __name__ == "__main__":
    main()