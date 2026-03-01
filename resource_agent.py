import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from dotenv import load_dotenv

# from googlegenai import AzureOpenAI
from google import genai

load_dotenv()

DEBUG = False
MODEL_ID = "gemini-2.5-flash-lite"
RES_JSON_PATH = "knowledge_base/resources.json"
SAVED_RES_PATH = "knowledge_base/saved_resources.json"
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
{
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
}

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
{
  "ranked": [
    {
      "title": "string",
      "link": "string",
      "source": "local|arxiv|semantic_scholar|google_scholar",
      "why": "short sentence"
    }
  ]
}

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
        self.client = genai.Client(api_key=os.environ.get("Gemini_API_Key"))
        self.resources_json = self._load_resources_json(RES_JSON_PATH)
        self.available_tags = self._load_available_tags(self.resources_json)
        self.saved_resources = self._load_saved_resources(SAVED_RES_PATH)

    def _load_resources_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("resources", [])

    def _load_available_tags(self, resources: List[Dict]) -> set:
        return {tag.lower() for r in resources for tag in r.get("tags", [])}

    def _load_saved_resources(self, path: str) -> Dict[str, List[Dict[str, Any]]]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("categories", {})
        except (json.JSONDecodeError, OSError):
            return {}

    def _persist_saved_resources(self):
        data = {"categories": self.saved_resources}
        os.makedirs(os.path.dirname(SAVED_RES_PATH), exist_ok=True)
        with open(SAVED_RES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # Make LLM Calls ------------------------------------------------

    def _generate(self, prompt: str):
        response = self.client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )
        if DEBUG:
            print(response.text)
        return response.text

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

    # Functions -----------------------------------------------------

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
        for r in self.resources_json:
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

    def save_resource(
        self, resource: Dict[str, Any], category: str, notes: str = ""
    ) -> str:
        category = (category or "Uncategorized").strip()
        if not category:
            category = "Uncategorized"
        entry = {
            "title": resource.get("title", "Untitled"),
            "link": resource.get("link", ""),
            "source": resource.get("source", ""),
            "tags": resource.get("tags", []),
            "notes": notes,
            "added_at": time.strftime("%Y-%m-%d"),
        }
        self.saved_resources.setdefault(category, [])
        if any(
            e.get("link") == entry["link"]
            and e.get("title") == entry["title"]
            for e in self.saved_resources[category]
        ):
            return "Resource already saved in this category."
        self.saved_resources[category].append(entry)
        self._persist_saved_resources()
        return "Resource saved."

    def list_categories(self) -> List[str]:
        return sorted(self.saved_resources.keys())

    def list_saved_resources(self, category: str = "") -> List[Dict[str, Any]]:
        if not category:
            resources = []
            for cat, items in self.saved_resources.items():
                for item in items:
                    resources.append({**item, "category": cat})
            return resources
        return [
            {**item, "category": category}
            for item in self.saved_resources.get(category, [])
        ]

    def delete_saved_resource(self, link: str, category: str) -> str:
        if category not in self.saved_resources:
            return "Category not found."
        before = len(self.saved_resources[category])
        self.saved_resources[category] = [
            r for r in self.saved_resources[category] if r.get("link") != link
        ]
        if len(self.saved_resources[category]) == before:
            return "Resource not found."
        if not self.saved_resources[category]:
            del self.saved_resources[category]
        self._persist_saved_resources()
        return "Resource deleted."

    def saved_resources_table(self, category: str = "") -> List[List[str]]:
        rows = []
        for r in self.list_saved_resources(category):
            rows.append(
                [
                    r.get("category", ""),
                    r.get("title", ""),
                    r.get("link", ""),
                    r.get("source", ""),
                    r.get("notes", ""),
                    r.get("added_at", ""),
                ]
            )
        return rows

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

    def run(self, query: str):
        analysis = self.analyze_query(query)
        if not analysis or not analysis.get("is_resource_request", False):
            return "I can help find resources. Ask for papers, tutorials, datasets, or tools and include your topic."

        max_results = int(analysis.get("max_results", 8))
        user_needs = analysis.get("user_needs", "Find relevant resources.")

        local_results = self.search_local_resources(analysis)
        online_results = []

        if analysis.get("use_online", True):
            arxiv_query = analysis.get("arxiv_query", "")
            scholar_query = analysis.get("semantic_scholar_query", "")
            google_scholar_query = analysis.get("google_scholar_query", "")
            online_results.extend(self.search_arxiv(arxiv_query, max_results=5))
            time.sleep(0.5)
            online_results.extend(
                self.search_semantic_scholar(scholar_query, max_results=5)
            )
            time.sleep(0.5)
            online_results.extend(
                self.search_google_scholar(google_scholar_query, max_results=5)
            )

        candidates = (local_results + online_results)[: max_results * 2]
        ranked = self.rank_resources(query, user_needs, candidates)

        if not ranked:
            return "No relevant resources found."

        lines = []
        for i, r in enumerate(ranked, start=1):
            lines.append(
                f"{i}. {r.get('title', 'Untitled')}\n"
                f"   Link: {r.get('link', '')}\n"
                f"   Source: {r.get('source', '')}\n"
                f"   Why: {r.get('why', '')}"
            )
        return "\n".join(lines)

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
            arxiv_query = analysis.get("arxiv_query", "")
            scholar_query = analysis.get("semantic_scholar_query", "")
            google_scholar_query = analysis.get("google_scholar_query", "")
            online_results.extend(self.search_arxiv(arxiv_query, max_results=5))
            time.sleep(0.5)
            online_results.extend(
                self.search_semantic_scholar(scholar_query, max_results=5)
            )
            time.sleep(0.5)
            online_results.extend(
                self.search_google_scholar(google_scholar_query, max_results=5)
            )

        candidates = (local_results + online_results)[: max_results * 2]
        ranked = self.rank_resources(query, user_needs, candidates)
        return {"message": "", "ranked": ranked}


# Main ------------------------------------------------------------------------


def main():
    res_agent = ResourceAgent()

    while True:
        query = input("Student's resource-related query: ")
        agent_response = res_agent.run(query)
        print("\n", agent_response)


if __name__ == "__main__":
    main()
