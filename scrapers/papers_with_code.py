"""
Papers with Code Scraper

Scrapes benchmark results from paperswithcode.com
"""

import asyncio
import re
from datetime import datetime
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


class PaperBenchmark(BaseModel):
    """Benchmark result from a paper"""
    task: str
    dataset: str
    metric: str
    value: str
    model: str
    paper_title: str
    paper_url: str
    code_url: Optional[str] = None
    year: Optional[int] = None


class PapersWithCodeScraper:
    """Scraper for Papers with Code"""
    
    BASE_URL = "https://paperswithcode.com"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    
    def __init__(self, timeout: int = 30):
        self.client = httpx.AsyncClient(
            headers=self.HEADERS,
            timeout=timeout,
            follow_redirects=True
        )
    
    async def close(self):
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def get_task_leaderboard(self, task: str) -> list[PaperBenchmark]:
        """Get leaderboard for a specific task"""
        # Task URL format: paperswithcode.com/task/{task-name}
        url = f"{self.BASE_URL}/task/{task}"
        
        response = await self.client.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "lxml")
        return self._parse_leaderboard(soup, task)
    
    async def search_paper(self, query: str) -> list[dict]:
        """Search for papers"""
        url = f"{self.BASE_URL}/search"
        params = {"q": query}
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "lxml")
        return self._parse_search_results(soup)
    
    def _parse_leaderboard(self, soup: BeautifulSoup, task: str) -> list[PaperBenchmark]:
        """Parse leaderboard table"""
        benchmarks = []
        
        # Find the main table
        table = soup.find("table", {"class": re.compile(r"leaderboard")})
        if not table:
            return benchmarks
        
        rows = table.find_all("tr")[1:]  # Skip header
        
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 4:
                # Extract data
                model_cell = cells[0]
                metric_cell = cells[1]
                value_cell = cells[2]
                
                model_name = model_cell.get_text().strip()
                metric_name = metric_cell.get_text().strip()
                value = value_cell.get_text().strip()
                
                # Get paper link
                paper_link = model_cell.find("a", {"href": re.compile(r"/paper/")})
                paper_url = ""
                if paper_link:
                    paper_url = self.BASE_URL + paper_link.get("href", "")
                
                benchmarks.append(PaperBenchmark(
                    task=task,
                    dataset=task,
                    metric=metric_name,
                    value=value,
                    model=model_name,
                    paper_title=model_name,
                    paper_url=paper_url
                ))
        
        return benchmarks
    
    def _parse_search_results(self, soup: BeautifulSoup) -> list[dict]:
        """Parse search results"""
        results = []
        
        # Find paper cards
        cards = soup.find_all("div", {"class": re.compile(r"paper-card")})
        
        for card in cards:
            title_elem = card.find("a", {"class": re.compile(r"title")})
            abstract_elem = card.find("p", {"class": re.compile(r"abstract")})
            
            if title_elem:
                results.append({
                    "title": title_elem.get_text().strip(),
                    "url": self.BASE_URL + title_elem.get("href", ""),
                    "abstract": abstract_elem.get_text().strip() if abstract_elem else ""
                })
        
        return results
    
    async def get_paper_details(self, paper_slug: str) -> dict:
        """Get detailed info about a paper"""
        url = f"{self.BASE_URL}/paper/{paper_slug}"
        
        response = await self.client.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "lxml")
        
        # Extract title
        title = ""
        title_elem = soup.find("h1")
        if title_elem:
            title = title_elem.get_text().strip()
        
        # Extract abstract
        abstract = ""
        abstract_elem = soup.find("div", {"class": re.compile(r"abstract")})
        if abstract_elem:
            abstract = abstract_elem.get_text().strip()
        
        # Extract benchmarks
        benchmarks = self._extract_paper_benchmarks(soup)
        
        # Extract code link
        code_link = ""
        code_elem = soup.find("a", {"href": re.compile(r"github\.com")})
        if code_elem:
            code_link = code_elem.get("href", "")
        
        return {
            "title": title,
            "abstract": abstract,
            "benchmarks": benchmarks,
            "code_url": code_link,
            "url": url
        }
    
    def _extract_paper_benchmarks(self, soup: BeautifulSoup) -> list[dict]:
        """Extract benchmark tables from paper page"""
        benchmarks = []
        
        # Find all benchmark sections
        tables = soup.find_all("table")
        
        for table in tables:
            rows = table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all(["td"])
                if len(cells) >= 3:
                    method = cells[0].get_text().strip()
                    metric = cells[1].get_text().strip()
                    value = cells[2].get_text().strip()
                    
                    benchmarks.append({
                        "method": method,
                        "metric": metric,
                        "value": value
                    })
        
        return benchmarks


# Common benchmark tasks on Papers with Code
TASKS = [
    "image-classification", "object-detection", "semantic-segmentation",
    "instance-segmentation", "question-answering", "summarization",
    "translation", "language-modeling", "text-classification",
    "named-entity-recognition", "sentiment-analysis",
    "machine-translation", "speech-recognition"
]


async def get_task_benchmarks(task: str) -> list[PaperBenchmark]:
    """Convenience function"""
    async with PapersWithCodeScraper() as scraper:
        return await scraper.get_task_leaderboard(task)


if __name__ == "__main__":
    import json
    import sys
    
    async def main():
        task = sys.argv[1] if len(sys.argv) > 1 else "image-classification"
        
        print(f"Fetching {task} leaderboard...")
        results = await get_task_benchmarks(task)
        
        print(f"Found {len(results)} results:")
        for b in results[:10]:
            print(f"  {b.model}: {b.value} ({b.metric})")
        
        # Save
        with open(f"data/{task}_leaderboard.json", "w") as f:
            json.dump([b.model_dump() for b in results], f, indent=2)
    
    asyncio.run(main())
