"""
Hugging Face Model Page Scraper

Scrapes model pages from Hugging Face to extract:
- Model metadata (name, author, downloads, likes)
- Benchmark performance data
- Model card information
"""

import asyncio
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


class BenchmarkScore(BaseModel):
    """Represents a single benchmark score"""
    name: str
    value: float
    dataset: str
    num_shots: Optional[int] = None
    raw_value: Optional[str] = None


class ModelMetadata(BaseModel):
    """Model metadata from HF page"""
    model_id: str
    author: str
    name: str
    last_modified: Optional[str] = None
    downloads: int = 0
    likes: int = 0
    tags: list[str] = Field(default_factory=list)
    language: Optional[str] = None
    license: Optional[str] = None
    
    
class ModelData(BaseModel):
    """Complete model data from scraping"""
    metadata: ModelMetadata
    benchmarks: list[BenchmarkScore] = Field(default_factory=list)
    pipeline_tag: Optional[str] = None
    model_card_summary: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            "model_id": self.metadata.model_id,
            "author": self.metadata.author,
            "name": self.metadata.name,
            "downloads": self.metadata.downloads,
            "likes": self.metadata.likes,
            "benchmarks": [
                {
                    "name": b.name,
                    "value": b.value,
                    "dataset": b.dataset,
                    "num_shots": b.num_shots
                }
                for b in self.benchmarks
            ],
            "scraped_at": self.scraped_at.isoformat()
        }


class HuggingFaceScraper:
    """Scraper for Hugging Face model pages"""
    
    BASE_URL = "https://huggingface.co"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Known benchmark names to look for
    BENCHMARK_PATTERNS = [
        r"mmlu", r"humaneval", r"mbpp", r"truthfulqa", r"hellaswag",
        r"winogrande", r"gsm8k", r" DROP", r" Squad", r" GLUE",
        r"big bench", r"mmlupro", r"agieval", r"bbh", r"arc"
    ]
    
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
    
    async def get_model_page(self, model_id: str) -> str:
        """Fetch the model page HTML"""
        url = f"{self.BASE_URL}/{model_id}"
        response = await self.client.get(url)
        response.raise_for_status()
        return response.text
    
    def parse_model_id(self, model_id: str) -> tuple[str, str]:
        """Parse model ID into author and name"""
        parts = model_id.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
        return "unknown", model_id
    
    def extract_metadata(self, soup: BeautifulSoup, model_id: str) -> ModelMetadata:
        """Extract basic model metadata"""
        author, name = self.parse_model_id(model_id)
        
        # Extract downloads
        downloads = 0
        download_elem = soup.find("span", {"data-testid": "download-count"})
        if download_elem:
            download_text = download_elem.get_text()
            downloads = self._parse_number(download_text)
        
        # Extract likes
        likes = 0
        like_elem = soup.find("span", {"data-testid": "like-count"})
        if like_elem:
            likes_text = like_elem.get_text()
            likes = self._parse_number(likes_text)
        
        # Extract tags
        tags = []
        tag_container = soup.find("div", {"class": re.compile(r"tags")})
        if tag_container:
            tag_elems = tag_container.find_all("a", {"class": re.compile(r"tag")})
            tags = [t.get_text().strip() for t in tag_elems]
        
        # Extract language
        language = None
        lang_elem = soup.find("span", {"data-testid": "language"})
        if lang_elem:
            language = lang_elem.get_text().strip()
        
        # Extract license
        license_elem = soup.find("a", {"class": re.compile(r"license")})
        license_text = license_elem.get_text().strip() if license_elem else None
        
        return ModelMetadata(
            model_id=model_id,
            author=author,
            name=name,
            downloads=downloads,
            likes=likes,
            tags=tags,
            language=language,
            license=license_text
        )
    
    def extract_benchmarks(self, soup: BeautifulSoup) -> list[BenchmarkScore]:
        """Extract benchmark scores from the page"""
        benchmarks = []
        
        # Look for benchmark sections
        benchmark_sections = soup.find_all(
            "div", 
            {"class": re.compile(r"benchmark|evaluation|metrics", re.I)}
        )
        
        # Also search for table rows with benchmark data
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    name_cell = cells[0].get_text().strip().lower()
                    value_cell = cells[1].get_text().strip()
                    
                    # Check if this looks like a benchmark
                    for pattern in self.BENCHMARK_PATTERNS:
                        if pattern.lower() in name_cell:
                            value = self._parse_benchmark_value(value_cell)
                            if value is not None:
                                # Try to extract num_shots
                                shots = self._extract_num_shots(name_cell)
                                benchmarks.append(BenchmarkScore(
                                    name=name_cell,
                                    value=value,
                                    dataset=name_cell,
                                    num_shots=shots,
                                    raw_value=value_cell
                                ))
                                break
        
        return benchmarks
    
    def extract_pipeline_tag(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the pipeline tag (e.g., 'text-generation', 'chat')"""
        # Look for pipeline tag in various locations
        pipeline_elem = soup.find(
            "a", 
            {"href": re.compile(r"/pipeline-tag/")}
        )
        if pipeline_elem:
            href = pipeline_elem.get("href", "")
            return href.split("/")[-1]
        
        # Also check for badge elements
        badge = soup.find("span", {"class": re.compile(r"pipeline")})
        if badge:
            return badge.get_text().strip()
        
        return None
    
    def _parse_number(self, text: str) -> int:
        """Parse numbers like '1.2M', '500K'"""
        text = text.strip().upper().replace(",", "")
        
        multipliers = {
            "K": 1_000,
            "M": 1_000_000,
            "B": 1_000_000_000
        }
        
        for suffix, mult in multipliers.items():
            if suffix in text:
                try:
                    return int(float(text.replace(suffix, "")) * mult)
                except:
                    pass
        
        try:
            return int(text)
        except:
            return 0
    
    def _parse_benchmark_value(self, text: str) -> Optional[float]:
        """Parse benchmark value like '75.3%' or '0.753'"""
        text = text.strip()
        
        # Remove percentage
        text = text.replace("%", "")
        
        try:
            return float(text)
        except:
            return None
    
    def _extract_num_shots(self, text: str) -> Optional[int]:
        """Extract number of shots from benchmark name"""
        patterns = [
            r"(\d+)[_-]shot",
            r"(\d+)[\s-]*shot",
            r"shot[_\s]*(\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return int(match.group(1))
        
        return None
    
    async def get_model_info(self, model_id: str) -> ModelData:
        """Scrape complete model information"""
        html = await self.get_model_page(model_id)
        soup = BeautifulSoup(html, "lxml")
        
        metadata = self.extract_metadata(soup, model_id)
        benchmarks = self.extract_benchmarks(soup)
        pipeline_tag = self.extract_pipeline_tag(soup)
        
        return ModelData(
            metadata=metadata,
            benchmarks=benchmarks,
            pipeline_tag=pipeline_tag
        )


async def scrape_model(model_id: str) -> ModelData:
    """Convenience function to scrape a single model"""
    async with HuggingFaceScraper() as scraper:
        return await scraper.get_model_info(model_id)


if __name__ == "__main__":
    import json
    
    async def main():
        import sys
        model_id = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b"
        
        print(f"Scraping {model_id}...")
        data = await scrape_model(model_id)
        
        print(f"\nModel: {data.metadata.name}")
        print(f"Author: {data.metadata.author}")
        print(f"Downloads: {data.metadata.downloads:,}")
        print(f"Likes: {data.metadata.likes:,}")
        
        if data.benchmarks:
            print(f"\nBenchmarks ({len(data.benchmarks)} found):")
            for b in data.benchmarks:
                print(f"  - {b.name}: {b.value}")
        else:
            print("\nNo benchmarks found on page")
        
        # Save to JSON
        with open(f"data/{model_id.replace('/', '_')}.json", "w") as f:
            json.dump(data.to_dict(), f, indent=2)
        print(f"\nSaved to data/{model_id.replace('/', '_')}.json")
    
    asyncio.run(main())
