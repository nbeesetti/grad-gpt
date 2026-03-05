import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from dotenv import load_dotenv

from openai import AzureOpenAI
from supabase import create_client, Client

load_dotenv()

DEBUG = False

MODEL_ID = "gradgpt-chat"
SUPABASE_RESOURCES_TABLE = "resources"

endpoint = "https://gradgpt-openai.openai.azure.com/"
deployment = "gradgpt-chat"
api_version = "2024-12-01-preview"
subscription_key = os.environ.get("Azure_API_Key")


SUPABASE_KNOWLEDGE_TABLE = "KnowledgeBase"
RESOURCE_AGENT_ID = "2"
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

# Prompts -----------------------------------------------------------------


RESOURCE_ANALYSIS_PROMPT = """
You are a research support assistant for Computer Science Master's students.

Analyze the student's query and return JSON only.

Rules:
- Return valid JSON only (no markdown).
- Keep values concise.
- If the query is not asking for resources, set is_resource_request to false.

Schema:
{{
  "is_resource_request": true/false,
  "user_needs": "short sentence",
  "topics": ["topic1", "topic2"],
  "keywords": ["keyword1", "keyword2"],
  "resource_types": ["paper", "book", "tutorial", "course", "dataset", "tool", "survey"],
  "constraints": ["recent", "classic", "open-access", "practical", "theory"],
  "use_online": true/false,
  "max_results": 8,
  "arxiv_query": "search string or empty",
  "semantic_scholar_query": "search string or empty",
  "google_scholar_query": "search string or empty"
}}

Student query:
{query}
"""


RESOURCE_RANKING_PROMPT = """
You are a research support assistant for Computer Science Master's students.

You may ONLY recommend resources provided below.
You must NEVER invent links or tools.

Your task:
- Select the most relevant resources
- Rank them by relevance to student's query
- Return at most 5
- Briefly explain why each is useful
- Always include the link EXACTLY as provided
- Return valid JSON only with this schema:
{{
  "ranked": [
    {{
      "title": "string",
      "link": "string",
      "source": "local|arxiv|semantic_scholar|google_scholar",
      "why": "short sentence"
    }}
  ]
}}

Student query:
{query}

USER NEEDS:
{user_needs}

AVAILABLE RESOURCES:
{resource_context}
"""


# Resource Agent Class --------------------------------------------------------


class ResourceAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        self.resources_list = self._load_resources_from_supabase()
        self.available_tags = self._load_available_tags(self.resources_list)

    def _load_resources_from_supabase(self) -> List[Dict[str, Any]]:
        try:
            sb = supabase
            response = (
                sb.table("KnowledgeBase")
                .select("*")
                .contains("agentIds", ["2"])
                .execute()
            )
            rows = response.data or []
            resources = []
            for r in rows:
                tags = r.get("tags", [])
                if isinstance(tags, str):
                    tags = json.loads(tags) if tags else []
                resources.append(
                    {
                        "id": r.get("id"),
                        "title": r.get("title", ""),
                        "description": r.get("content", ""),
                        "url": r.get("sourceURL", ""),
                        "tags": tags,
                    }
                )
            return resources
        except Exception as e:
            if DEBUG:
                print(f"Supabase load error: {e}")
            return []

    def _load_available_tags(self, resources: List[Dict]) -> set:
        return {tag.lower() for r in resources for tag in r.get("tags", [])}

    # LLM Call ----------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        if DEBUG:
            print(text)
        return text

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return {}
            return {}

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    # Core Functions ----------------------------------------------------------

    def analyze_query(self, query: str) -> Dict[str, Any]:
        prompt = RESOURCE_ANALYSIS_PROMPT.format(query=query)
        response = self._generate(prompt)
        analysis = self._safe_json_loads(response)
        return analysis if isinstance(analysis, dict) else {}

    def _score_resource(self, r: Dict[str, Any], keywords: List[str]) -> float:
        if not keywords:
            return 0.0
        text = self._normalize(
            f"{r.get('title', '')} {r.get('description', '')} {' '.join(r.get('tags', []))}"
        )
        score = 0.0
        for kw in keywords:
            if self._normalize(kw) in text:
                score += 1.0
        return score

    def search_local_resources(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        keywords = analysis.get("keywords", []) + analysis.get("topics", [])
        ranked = []
        for r in self.resources_list:
            score = self._score_resource(r, keywords)
            if score > 0:
                ranked.append((score, r))
        ranked.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, r in ranked[:15]:
            results.append(
                {
                    "title": r.get("title", ""),
                    "description": r.get("description", ""),
                    "link": r.get("url", ""),
                    "tags": r.get("tags", []),
                    "source": "local",
                    "score": score,
                }
            )
        return results

    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not query:
            return []
        encoded = urllib.parse.quote(query)
        url = (
            "http://export.arxiv.org/api/query?"
            f"search_query={encoded}&start=0&max_results={max_results}"
        )
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                xml_data = response.read()
        except Exception:
            return []
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError:
            return []
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results = []
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", default="", namespaces=ns).strip()
            summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
            link = entry.findtext("atom:id", default="", namespaces=ns).strip()
            results.append(
                {
                    "title": title,
                    "description": summary,
                    "link": link,
                    "tags": ["arxiv"],
                    "source": "arxiv",
                    "score": 0.0,
                }
            )
        return results

    def search_semantic_scholar(
        self, query: str, max_results: int = 5
    ) -> List[Dict[str, Any]]:
        if not query:
            return []
        params = urllib.parse.urlencode(
            {
                "query": query,
                "limit": max_results,
                "fields": "title,abstract,url,authors,year,venue",
            }
        )
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []
        results = []
        for item in data.get("data", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "description": item.get("abstract", "") or "",
                    "link": item.get("url", ""),
                    "tags": ["semantic_scholar"],
                    "source": "semantic_scholar",
                    "score": 0.0,
                }
            )
        return results

    def search_google_scholar(
        self, query: str, max_results: int = 5
    ) -> List[Dict[str, Any]]:
        if not query:
            return []
        api_key = os.environ.get("SERPAPI_API_KEY") or os.environ.get(
            "GOOGLE_SCHOLAR_API_KEY"
        )
        if not api_key:
            return []
        params = urllib.parse.urlencode(
            {
                "engine": "google_scholar",
                "q": query,
                "api_key": api_key,
                "num": max_results,
            }
        )
        url = f"{SERPAPI_ENDPOINT}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []
        results = []
        for item in data.get("organic_results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "description": item.get("snippet", "") or "",
                    "link": item.get("link", ""),
                    "tags": ["google_scholar"],
                    "source": "google_scholar",
                    "score": 0.0,
                }
            )
        return results

    def rank_resources(
        self, query: str, user_needs: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        resource_context = []
        for i, r in enumerate(candidates):
            resource_context.append(
                f"[{i+1}]\nTitle: {r['title']}\nDescription: {r['description']}\nLink: {r['link']}\nSource: {r['source']}\nTags: {', '.join(r.get('tags', []))}"
            )
        context = "\n\n".join(resource_context)
        prompt = RESOURCE_RANKING_PROMPT.format(
            query=query, user_needs=user_needs, resource_context=context
        )
        response = self._generate(prompt)
        parsed = self._safe_json_loads(response)
        ranked = parsed.get("ranked", []) if isinstance(parsed, dict) else []
        if ranked:
            return ranked
        # Fallback: return top candidates by heuristic score
        return [
            {
                "title": r.get("title", ""),
                "link": r.get("link", ""),
                "source": r.get("source", ""),
                "why": "Matched by keyword overlap with the query.",
            }
            for r in candidates[:5]
        ]

    def _format_response_for_chat(self, ranked: List[Dict[str, Any]]) -> str:
        lines = ["## Recommended resources\n"]
        for i, r in enumerate(ranked, start=1):
            title = r.get("title", "Untitled")
            link = r.get("link", "")
            source = r.get("source", "")
            why = r.get("why", "")
            link_md = f"[{title}]({link})" if link else title
            lines.append(f"**{i}. {link_md}**  \n_{source}_")
            if why:
                lines.append(f"   {why}")
            lines.append("")
        return "\n".join(lines).strip()

    def run(self, query: str) -> str:
        analysis = self.analyze_query(query)
        if not analysis or not analysis.get("is_resource_request", False):
            return "I can help find resources. Ask for papers, tutorials, datasets, or tools and include your topic."

        max_results = int(analysis.get("max_results", 8))
        user_needs = analysis.get("user_needs", "Find relevant resources.")

        local_results = self.search_local_resources(analysis)
        online_results = []

        if analysis.get("use_online", True):
            online_results.extend(
                self.search_arxiv(analysis.get("arxiv_query", ""), max_results=5)
            )
            time.sleep(0.5)
            online_results.extend(
                self.search_semantic_scholar(
                    analysis.get("semantic_scholar_query", ""), max_results=5
                )
            )
            time.sleep(0.5)
            online_results.extend(
                self.search_google_scholar(
                    analysis.get("google_scholar_query", ""), max_results=5
                )
            )

        candidates = (local_results + online_results)[: max_results * 2]
        ranked = self.rank_resources(query, user_needs, candidates)

        if not ranked:
            return "No relevant resources found."

        return self._format_response_for_chat(ranked)

    def run_structured(self, query: str) -> Dict[str, Any]:
        analysis = self.analyze_query(query)
        if not analysis or not analysis.get("is_resource_request", False):
            return {
                "message": "Ask for resources (papers, tutorials, datasets, tools) and include your topic.",
                "ranked": [],
            }

        max_results = int(analysis.get("max_results", 8))
        user_needs = analysis.get("user_needs", "Find relevant resources.")

        local_results = self.search_local_resources(analysis)
        online_results = []

        if analysis.get("use_online", True):
            online_results.extend(
                self.search_arxiv(analysis.get("arxiv_query", ""), max_results=5)
            )
            time.sleep(0.5)
            online_results.extend(
                self.search_semantic_scholar(
                    analysis.get("semantic_scholar_query", ""), max_results=5
                )
            )
            time.sleep(0.5)
            online_results.extend(
                self.search_google_scholar(
                    analysis.get("google_scholar_query", ""), max_results=5
                )
            )

        candidates = (local_results + online_results)[: max_results * 2]
        ranked = self.rank_resources(query, user_needs, candidates)
        message = (
            self._format_response_for_chat(ranked)
            if ranked
            else "No relevant resources found."
        )
        return {"message": message, "ranked": ranked}


# Main ------------------------------------------------------------------------


def main():
    res_agent = ResourceAgent()
    while True:
        query = input("Student's resource-related query: ")
        agent_response = res_agent.run(query)
        print("\n", agent_response)


if __name__ == "__main__":
    main()
