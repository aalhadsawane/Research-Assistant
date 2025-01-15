from typing import List, Dict
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool
from tavily import TavilyClient
from pydantic import BaseModel

# Load environment variables
load_dotenv()

class ResearchResult(BaseModel):
    final_summary: str
    sources: List[Dict]
    iteration_summaries: List[str]

class ResearchAgent:
    def __init__(self, tavily_api_key: str):
        # Initialize LLM using Ollama
        self.llm = Ollama(
            model="llama3.2",
            temperature=0.3,
        )
        
        # Initialize Tavily client with API key in headers
        self.tavily_api_key = tavily_api_key
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        
        # Create query generation chain
        query_template = """Generate a short and focused web search query (maximum 10 words) based on the following:
        Research Topic: {research_topic}
        Previous Findings: {previous_findings}
        Research Gaps: {gaps}
        
        Make the query general enough to find relevant results but specific enough to stay on topic.
        Focus on key concepts and avoid overly specific constraints.
        
        Return ONLY the search query without any explanation or additional text."""
        
        self.query_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=query_template,
                input_variables=["research_topic", "previous_findings", "gaps"]
            )
        )
        
        # Create summary chain
        summary_template = """Analyze the following search results and create a comprehensive summary.
        The summary should be approximately {target_word_count} words.
        Format the summary using markdown with the following structure:
        
        # Overview
        [Brief overview of the topic]
        
        # Key Findings
        [Main points and insights]
        
        # Analysis
        [Detailed analysis of the findings]
        
        # Conclusion
        [Summary of conclusions and implications]
        
        Use proper markdown formatting with headings (using # for headings), bullet points (using - for lists), and emphasis (using ** for bold text).
        Ensure all headings are plain text without any special characters or formatting.
        
        Search Results: {search_results}
        Previous Summary: {previous_summary}
        
        Summary:
        
        Research Gaps:"""
        
        self.summary_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=summary_template,
                input_variables=["search_results", "previous_summary", "target_word_count"]
            )
        )

    def search_web(self, query: str) -> List[Dict]:
        """Perform web search using Tavily"""
        try:
            print(f"\nMaking Tavily search request:")
            print(f"Query: {query}")
            
            # Most basic search possible
            search_results = self.tavily_client.search(
                query=query,
                include_answer=False,
                include_domains=[],
                exclude_domains=[],
                max_results=5
            )
            
            if isinstance(search_results, dict):
                results = search_results.get('results', [])
                print(f"Found {len(results)} results")
                if not results:
                    print(f"No results found for query")
                return results
            print(f"Unexpected response type: {type(search_results)}")
            return []
        except Exception as e:
            import traceback
            print(f"Tavily search error: {str(e)}")
            print("Full error traceback:")
            print(traceback.format_exc())
            return []
    
    def generate_query(self, topic: str, previous_findings: str = "", gaps: str = "") -> str:
        """Generate a search query based on the topic and previous findings"""
        result = self.query_chain.invoke({
            "research_topic": topic,
            "previous_findings": previous_findings,
            "gaps": gaps
        })
        return result["text"].strip()
    
    def create_summary(self, search_results: List[Dict], previous_summary: str = "", target_word_count: int = 300) -> Dict:
        """Create a summary from search results and identify gaps"""
        formatted_results = "\n".join([
            f"Title: {result.get('title', '')}\nContent: {result.get('content', '')}\n"
            for result in search_results
        ])
        
        result = self.summary_chain.invoke({
            "search_results": formatted_results,
            "previous_summary": previous_summary,
            "target_word_count": target_word_count
        })
        
        # Split into summary and gaps
        parts = result["text"].split("Research Gaps:")
        summary = parts[0].strip()
        gaps = parts[1].strip() if len(parts) > 1 else ""
        
        return {"summary": summary, "gaps": gaps}
    
    def research(self, topic: str, iterations: int = 3, target_word_count: int = 300) -> ResearchResult:
        """Main research function that performs iterative research"""
        all_sources = []
        summaries = []
        current_summary = ""
        current_gaps = ""
        
        for i in range(iterations):
            # Generate search query
            query = self.generate_query(topic, current_summary, current_gaps)
            
            # Perform web search
            search_results = self.search_web(query)
            if search_results:
                all_sources.extend(search_results)
                
                # Create summary and identify gaps
                result = self.create_summary(search_results, current_summary, target_word_count)
                current_summary = result["summary"]
                current_gaps = result["gaps"]
                # Store both summary and its sources
                summaries.append({
                    "summary": current_summary,
                    "sources": search_results
                })
            else:
                print(f"No search results found for iteration {i+1}")
                # Stop further iterations if no results found
                break
        
        # If we have no summaries at all, return error state
        if not summaries:
            return ResearchResult(
                final_summary="No results found for the given topic.",
                sources=[],
                iteration_summaries=[]
            )
        
        return ResearchResult(
            final_summary=summaries[-1]["summary"],  # Use last summary as final
            sources=all_sources,
            iteration_summaries=[s["summary"] for s in summaries]
        )

def main():
    # Load environment variables
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if not tavily_api_key:
        raise ValueError("Please set TAVILY_API_KEY environment variables")
    
    # Initialize agent
    agent = ResearchAgent(tavily_api_key=tavily_api_key)
    
    # Example usage
    topic = input("Enter your research topic: ")
    iterations = int(input("Enter number of research iterations: "))
    
    result = agent.research(topic, iterations)
    
    print("\nFinal Summary:")
    print(result.final_summary)
    print("\nSources:")
    for source in result.sources:
        print(f"- {source['title']}: {source['url']}")

if __name__ == "__main__":
    main() 