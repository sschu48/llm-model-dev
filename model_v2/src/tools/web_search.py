# if documents aren't relevant, then model will search the web
# will return findings like document

import os
from typing import List, Dict, Any
from tavily import TavilyClient

class WebSearcher:
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable is not set")
        self.client = TavilyClient(api_key=self.api_key)

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )
            
            # Extract relevant information from the response
            results = []
            for result in response['results']:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'score': result.get('score', 0)
                })
            
            return results
        except Exception as e:
            print(f"Error performing Tavily search: {e}")
            return []

    def get_search_context(self, query: str, max_results: int = 5) -> str:
        results = self.search(query, max_results)
        context = f"Web search results for query: '{query}'\n\n"
        for i, result in enumerate(results, 1):
            context += f"{i}. {result['title']}\n"
            context += f"   URL: {result['url']}\n"
            context += f"   Summary: {result['content'][:200]}...\n\n"
        return context