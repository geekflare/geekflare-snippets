import os
import sys
import json
import requests
import anthropic
from datetime import datetime
from dotenv import load_dotenv


sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GEEKFLARE_API_KEY = os.getenv("GEEKFLARE_API_KEY")

if not CLAUDE_API_KEY or not GEEKFLARE_API_KEY:
    raise RuntimeError(
        "Missing CLAUDE_API_KEY or GEEKFLARE_API_KEY. Add them to your .env file."
    )


claude_client = anthropic.Anthropic(
    api_key=CLAUDE_API_KEY
)

MODEL = "claude-opus-5"


research_question = """
What are the latest developments in AI agents for business,
and how are companies using them?
"""


def generate_search_queries(question):
    prompt = f"""
You are a web research agent.

Break the following research question into 3 to 5 focused
web search queries.

Research question:
{question}
"""

    response = claude_client.messages.create(
        model=MODEL,
        max_tokens=1024,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["queries"],
                    "additionalProperties": False
                }
            }
        },
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["queries"]


def search_web(query):
    url = "https://api.geekflare.com/search"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": GEEKFLARE_API_KEY
    }

    payload = {
        "query": query
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    response.raise_for_status()

    return response.json()


def review_results(question, results):
    research_data = json.dumps(
        results,
        indent=2
    )

    prompt = f"""
You are reviewing web research results.

Original research question:
{question}

Research results:
{research_data}

Check if the available information is enough to answer
the research question. If information is missing, set
"enough_information" to false and add 1 to 3 additional
search queries. Otherwise return an empty list for
"additional_queries".
"""

    response = claude_client.messages.create(
        model=MODEL,
        max_tokens=1024,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "enough_information": {"type": "boolean"},
                        "additional_queries": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["enough_information", "additional_queries"],
                    "additionalProperties": False
                }
            }
        },
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def generate_final_answer(question, results):
    research_data = json.dumps(
        results,
        indent=2
    )

    prompt = f"""
You are a web research agent.

Answer the following research question using the
research results provided below.

Research question:
{question}

Research results:
{research_data}

Write a clear and structured answer based on the
available information.

Add the source URLs used for the research.
"""

    response = claude_client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return next(b.text for b in response.content if b.type == "text")


def main():
    search_queries = generate_search_queries(research_question)

    print("Search Queries:")
    for query in search_queries:
        print("-", query)

    all_results = []

    for query in search_queries:
        print(f"\nSearching: {query}")

        search_response = search_web(query)

        all_results.append({
            "query": query,
            "results": search_response
        })

    print("\nSearch completed.")

    review = review_results(research_question, all_results)

    print("\nResearch Review:")
    print(review)

    if not review["enough_information"]:
        additional_queries = review["additional_queries"]

        print("\nAdditional Searches:")

        for query in additional_queries:
            print("-", query)

            search_response = search_web(query)

            all_results.append({
                "query": query,
                "results": search_response
            })

    final_answer = generate_final_answer(research_question, all_results)

    print("\nFinal Research Answer:\n")
    print(final_answer)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"research_{timestamp}.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Research Question\n\n{research_question.strip()}\n\n")
        f.write(final_answer)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
