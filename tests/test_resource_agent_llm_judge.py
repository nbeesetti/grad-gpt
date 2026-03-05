import json
import os
import re
from typing import Dict, Any

import pytest
from openai import AzureOpenAI

from resource_agent import ResourceAgent

pytestmark = pytest.mark.skipif(
    not os.environ.get("Azure_API_Key"),
    reason="LLM judge tests require Azure_API_Key",
)

AZURE_ENDPOINT = "https://gradgpt-openai.openai.azure.com/"
AZURE_DEPLOYMENT = "gradgpt-chat"
AZURE_API_VERSION = "2024-12-01-preview"

RESPONSE_RELEVANCE_JUDGE_PROMPT = """
You are a judge evaluating whether a resource-agent response appropriately answers a user's query.

QUERY: {query}

RESPONSE: {response}

Evaluate:
1. RELEVANCE: Does the response address what the user asked for?
2. HELPFULNESS: Would a grad student find this useful?
3. FORMAT: Are resources clearly presented with titles and links (when applicable)?
4. GROUNDING: If resources are listed, do they seem relevant to the query topic (no hallucinated links)?

Return JSON only:
{{"pass": true/false, "score": 1-5, "reason": "one sentence explanation"}}

Be strict: pass=false if the response is off-topic, empty, or recommends irrelevant resources.
"""

NON_RESOURCE_QUERY_JUDGE_PROMPT = """
You are a judge evaluating how the resource agent handles a non-resource query.

QUERY: {query}
(This query is NOT asking for papers, tutorials, tools, or other resources.)

RESPONSE: {response}

The agent should NOT recommend resources. It should redirect or clarify.

Return JSON only:
{{"pass": true/false, "reason": "one sentence"}}

pass=true if the response appropriately does NOT list resources and invites the user to ask for resources.
pass=false if the response wrongly recommends resources or is confusing.
"""

ACCURACY_JUDGE_PROMPT = """
You are a judge evaluating the accuracy and trustworthiness of a resource-agent response.

QUERY: {query}

RESPONSE: {response}

Evaluate:
1. ACCURACY: Do the resource titles and descriptions match the query topic?
2. NO HALLUCINATION: Are links plausible and not obviously fabricated?
3. SPECIFICITY: Are the resources specific to the query rather than generic?

Return JSON only:
{{"pass": true/false, "score": 1-5, "reason": "one sentence explanation"}}

pass=false if resources appear fabricated, are completely off-topic, or descriptions clearly don't match titles.
"""

COMPLETENESS_JUDGE_PROMPT = """
You are a judge evaluating whether a resource-agent response is complete and sufficiently covers the user's needs.

QUERY: {query}

RESPONSE: {response}

Evaluate:
1. COVERAGE: Does the response cover the main aspects of what was asked?
2. QUANTITY: Are enough resources provided (at least 2-3 for a resource request)?
3. DIVERSITY: Are different types of resources included when appropriate (papers, tools, tutorials)?

Return JSON only:
{{"pass": true/false, "score": 1-5, "reason": "one sentence explanation"}}
"""


# Helpers ---------------------------------------------------------------------


def _safe_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start >= 0:
        depth, end = 0, start
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def _get_judge_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=os.environ.get("Azure_API_Key"),
    )


def llm_judge(query: str, response: str, judge_prompt: str) -> Dict[str, Any]:
    client = _get_judge_client()
    prompt = judge_prompt.format(query=query, response=response)
    out = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
    )
    return _safe_json(out.choices[0].message.content or "")


# Unit tests (no judge) -------------------------------------------------------


def test_analysis_is_resource_request():
    agent = ResourceAgent()
    analysis = agent.analyze_query(
        "I need recent papers on RAG evaluation benchmarks for LLMs."
    )
    assert isinstance(analysis, dict)
    assert analysis.get("is_resource_request") is True
    assert analysis.get("arxiv_query")


def test_analysis_non_resource_query():
    agent = ResourceAgent()
    analysis = agent.analyze_query("What is the deadline for thesis submission?")
    assert isinstance(analysis, dict)
    assert analysis.get("is_resource_request") is False


def test_ranking_returns_expected_fields():
    agent = ResourceAgent()
    candidates = [
        {
            "title": "RAG Evaluation Survey",
            "description": "Survey of evaluation methods for RAG systems.",
            "link": "https://example.com/rag",
            "source": "arxiv",
            "tags": ["rag", "evaluation"],
        },
        {
            "title": "Dataset for LLM Benchmarks",
            "description": "Benchmark dataset for LLM evaluation.",
            "link": "https://example.com/benchmarks",
            "source": "semantic_scholar",
            "tags": ["benchmark"],
        },
    ]
    ranked = agent.rank_resources(
        "Find RAG evaluation papers.", "RAG evaluation resources.", candidates
    )
    assert ranked
    assert "title" in ranked[0]
    assert "link" in ranked[0]
    assert "why" in ranked[0]


def test_ranking_does_not_hallucinate_links():
    """Ranked results should only contain links from the candidates."""
    agent = ResourceAgent()
    known_links = {"https://example.com/rag", "https://example.com/benchmarks"}
    candidates = [
        {
            "title": "RAG Evaluation Survey",
            "description": "Survey of evaluation methods for RAG systems.",
            "link": "https://example.com/rag",
            "source": "arxiv",
            "tags": ["rag"],
        },
        {
            "title": "LLM Benchmark Dataset",
            "description": "Benchmark dataset for LLM evaluation.",
            "link": "https://example.com/benchmarks",
            "source": "semantic_scholar",
            "tags": ["benchmark"],
        },
    ]
    ranked = agent.rank_resources(
        "RAG evaluation papers.", "RAG resources.", candidates
    )
    for r in ranked:
        assert (
            r.get("link") in known_links
        ), f"Hallucinated link detected: {r.get('link')}"


def test_run_non_resource_returns_redirect():
    agent = ResourceAgent()
    response = agent.run("What is the deadline for thesis submission?")
    assert "resource" in response.lower() or "ask" in response.lower()
    assert "http" not in response


def test_run_structured_returns_expected_shape():
    agent = ResourceAgent()
    result = agent.run_structured("Find me papers on transformer architectures.")
    assert "message" in result
    assert "ranked" in result
    assert isinstance(result["ranked"], list)


# LLM-as-judge tests ----------------------------------------------------------


def test_judge_resource_request_relevance():
    """Resource query should get relevant, well-formatted recommendations."""
    agent = ResourceAgent()
    query = (
        "I need tools and sites to find research papers and literature for my thesis."
    )
    response = agent.run(query)
    verdict = llm_judge(query, response, RESPONSE_RELEVANCE_JUDGE_PROMPT)
    assert verdict.get("pass") is True, verdict.get("reason", "No reason")
    assert verdict.get("score", 0) >= 3, f"Score too low: {verdict}"


def test_judge_academic_writing_resources():
    """Query about academic writing should return relevant resources."""
    agent = ResourceAgent()
    query = "Find me resources for academic writing and literature review."
    response = agent.run(query)
    verdict = llm_judge(query, response, RESPONSE_RELEVANCE_JUDGE_PROMPT)
    assert verdict.get("pass") is True, verdict.get("reason", "No reason")


def test_judge_non_resource_query_redirected():
    """Non-resource query should get a redirect, not fake recommendations."""
    agent = ResourceAgent()
    query = "What is the deadline for thesis submission?"
    response = agent.run(query)
    verdict = llm_judge(query, response, NON_RESOURCE_QUERY_JUDGE_PROMPT)
    assert verdict.get("pass") is True, verdict.get("reason", "No reason")


def test_judge_why_explanations_present_and_useful():
    """Each ranked resource should have a meaningful 'why' explanation."""
    agent = ResourceAgent()
    query = "I need papers and tools for my CS research."
    result = agent.run_structured(query)
    ranked = result.get("ranked", [])
    if not ranked:
        pytest.skip("No ranked results; online APIs may be unavailable")
    verdict = llm_judge(
        query, result.get("message", ""), RESPONSE_RELEVANCE_JUDGE_PROMPT
    )
    assert verdict.get("pass") is True, verdict.get("reason", "No reason")
    for r in ranked:
        assert r.get("why"), f"Resource '{r.get('title')}' missing 'why' explanation"


def test_judge_deep_learning_resources_relevant():
    """Query with specific ML topic should get topic-relevant resources."""
    agent = ResourceAgent()
    query = "Recommend papers and tools for deep learning and neural networks."
    response = agent.run(query)
    verdict = llm_judge(query, response, RESPONSE_RELEVANCE_JUDGE_PROMPT)
    assert verdict.get("pass") is True, verdict.get("reason", "No reason")


def test_judge_response_format_quality():
    """Response should be clearly formatted with markdown for chat display."""
    agent = ResourceAgent()
    query = "I'm looking for research paper databases and citation tools."
    response = agent.run(query)
    verdict = llm_judge(query, response, RESPONSE_RELEVANCE_JUDGE_PROMPT)
    assert verdict.get("pass") is True, verdict.get("reason", "No reason")
    if "No relevant" not in response:
        assert (
            "http" in response or "##" in response or "**" in response
        ), "Response with resources should contain links or markdown formatting"


def test_judge_accuracy_no_hallucinated_resources():
    """Resources returned should be accurate and not hallucinated."""
    agent = ResourceAgent()
    query = "Find survey papers on natural language processing."
    response = agent.run(query)
    verdict = llm_judge(query, response, ACCURACY_JUDGE_PROMPT)
    assert verdict.get("pass") is True, verdict.get("reason", "No reason")
    assert verdict.get("score", 0) >= 3, f"Accuracy score too low: {verdict}"


def test_judge_completeness_enough_resources():
    """Response to a broad resource query should cover enough ground."""
    agent = ResourceAgent()
    query = "I need papers, tools, and datasets for my machine learning thesis."
    response = agent.run(query)
    verdict = llm_judge(query, response, COMPLETENESS_JUDGE_PROMPT)
    assert verdict.get("pass") is True, verdict.get("reason", "No reason")
    assert verdict.get("score", 0) >= 3, f"Completeness score too low: {verdict}"


def test_judge_specificity_targeted_query():
    """A specific query should return specific, not generic, resources."""
    agent = ResourceAgent()
    query = (
        "I need resources specifically about graph neural networks for drug discovery."
    )
    response = agent.run(query)
    verdict = llm_judge(query, response, ACCURACY_JUDGE_PROMPT)
    assert verdict.get("pass") is True, verdict.get("reason", "No reason")


def test_judge_vague_resource_query_still_helpful():
    """Even a vague resource query should produce a helpful response."""
    agent = ResourceAgent()
    query = "I need help with my research."
    response = agent.run(query)
    # Vague queries may redirect — both redirect and resource responses are valid
    relevance_verdict = llm_judge(query, response, RESPONSE_RELEVANCE_JUDGE_PROMPT)
    non_resource_verdict = llm_judge(query, response, NON_RESOURCE_QUERY_JUDGE_PROMPT)
    assert (
        relevance_verdict.get("pass") is True
        or non_resource_verdict.get("pass") is True
    ), f"Vague query handled poorly. Relevance: {relevance_verdict}, Non-resource: {non_resource_verdict}"
