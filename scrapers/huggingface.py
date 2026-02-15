"""
HuggingFace Model Scraper

Scrapes model metadata, downloads, likes, and performance metrics from HuggingFace.
"""

import asyncio
import re
from datetime import datetime
from typing import Optional
import httpx
from pydantic import BaseModel, Field


class HuggingFaceModel(BaseModel):
    """HuggingFace model data"""
    model_id: str
    author: str
    name: str
    sha: str = ""
    last_modified: str = ""
    private: bool = False
    downloads: int = 0
    likes: int = 0
    tags: list[str] = Field(default_factory=list)
    pipeline_tag: Optional[str] = None
    created_at: str = ""
    siblings: list[dict] = Field(default_factory=list)
    
    def to_dict(self):
        return self.model_dump()


class ModelCard(BaseModel):
    """Model card data"""
    model_id: str
    language: Optional[str] = None
    license: Optional[str] = None
    library_name: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    datasets: list[str] = Field(default_factory=list)
    pipelines: list[str] = Field(default_factory=list)


class HuggingFaceScraper:
    """Scraper for HuggingFace models"""
    
    BASE_URL = "https://huggingface.co"
    API_URL = "https://huggingface.co/api"
    
    def __init__(self, timeout: int = 30):
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def close(self):
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def get_model(self, model_id: str) -> HuggingFaceModel:
        """Get model metadata"""
        url = f"{self.API_URL}/models/{model_id}"
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        
        return HuggingFaceModel(
            model_id=model_id,
            author=data.get("author", ""),
            name=data.get("modelId", ""),
            sha=data.get("sha", ""),
            last_modified=data.get("lastModified", ""),
            private=data.get("private", False),
            downloads=data.get("downloads", 0),
            likes=data.get("likes", 0),
            tags=data.get("tags", []),
            pipeline_tag=data.get("pipeline_tag"),
            created_at=data.get("createdAt", ""),
            siblings=data.get("siblings", [])
        )
    
    async def get_model_card(self, model_id: str) -> ModelCard:
        """Get model card (README) data"""
        url = f"{self.API_URL}/models/{model_id}"
        params = {"repo": "model"}
        response = await self.client.get(url, params=params)
        
        card = ModelCard(model_id=model_id)
        
        # Try to get model card from repo
        readme_url = f"{self.BASE_URL}/{model_id}/raw/main/README.md"
        try:
            response = await self.client.get(readme_url)
            if response.status_code == 200:
                content = response.text
                # Parse YAML frontmatter
                card = self._parse_readme(model_id, content)
        except:
            pass
        
        return card
    
    def _parse_readme(self, model_id: str, content: str) -> ModelCard:
        """Parse README to extract model card data"""
        card = ModelCard(model_id=model_id)
        
        # Extract tags
        tag_pattern = r'tags:\s*\n((?:\s*-\s*.+\n)+)'
        tags_match = re.search(tag_pattern, content)
        if tags_match:
            tags = re.findall(r'-\s*(.+)', tags_match.group(1))
            card.tags = [t.strip() for t in tags]
        
        # Extract language
        lang_pattern = r'language:\s*(\w+)'
        lang_match = re.search(lang_pattern, content)
        if lang_match:
            card.language = lang_match.group(1)
        
        # Extract license
        license_pattern = r'license:\s*(\w+)'
        license_match = re.search(license_pattern, content)
        if license_match:
            card.license = license_match.group(1)
        
        # Extract library
        lib_pattern = r'library_name:\s*(\w+)'
        lib_match = re.search(lib_pattern, content)
        if lib_match:
            card.library_name = lib_match.group(1)
        
        return card
    
    async def search_models(
        self, 
        query: str = "",
        sort: str = "downloads",
        direction: str = -1,
        limit: int = 100
    ) -> list[HuggingFaceModel]:
        """Search models"""
        url = f"{self.API_URL}/models"
        params = {
            "search": query,
            "sort": sort,
            "direction": direction,
            "limit": limit
        }
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        
        models = []
        for data in response.json():
            model_id = data.get("id", "")
            models.append(HuggingFaceModel(
                model_id=model_id,
                author=data.get("author", ""),
                name=data.get("modelId", ""),
                sha=data.get("sha", ""),
                last_modified=data.get("lastModified", ""),
                private=data.get("private", False),
                downloads=data.get("downloads", 0),
                likes=data.get("likes", 0),
                tags=data.get("tags", []),
                pipeline_tag=data.get("pipeline_tag"),
                created_at=data.get("createdAt", ""),
                siblings=data.get("siblings", [])
            ))
        
        return models
    
    async def get_trending_models(
        self, 
        timeframe: str = "trending",
        limit: int = 100
    ) -> list[HuggingFaceModel]:
        """Get trending models"""
        return await self.search_models(
            query=timeframe,
            sort="downloads",
            limit=limit
        )
    
    async def get_models_by_task(
        self, 
        task: str,
        limit: int = 50
    ) -> list[HuggingFaceModel]:
        """Get models by task/pipeline"""
        return await self.search_models(
            query=f"task:{task}",
            sort="downloads",
            limit=limit
        )


# Example tasks to scrape
POPULAR_TASKS = [
    "text-generation",
    "text-classification", 
    "token-classification",
    "question-answering",
    "summarization",
    "translation",
    "image-classification",
    "object-detection",
    "image-segmentation",
    "text-to-image",
    "image-to-text",
    "automatic-speech-recognition",
    "text-to-speech",
    "feature-extraction",
    "embeddings",
]


async def scrape_task_models():
    """Scrape models by task"""
    async with HuggingFaceScraper() as scraper:
        all_models = {}
        
        for task in POPULAR_TASKS:
            print(f"Scraping {task}...")
            models = await scraper.get_models_by_task(task, limit=20)
            all_models[task] = [m.model_dump() for m in models]
            print(f"  Found {len(models)} models")
        
        return all_models


async def scrape_top_models(limit: int = 500):
    """Scrape top models"""
    async with HuggingFaceScraper() as scraper:
        models = await scraper.get_trending_models(limit=limit)
        return [m.model_dump() for m in models]


if __name__ == "__main__":
    import json
    import sys
    
    async def main():
        if len(sys.argv) > 1:
            # Scrape specific model
            model_id = sys.argv[1]
            async with HuggingFaceScraper() as scraper:
                model = await scraper.get_model(model_id)
                card = await scraper.get_model_card(model_id)
                
                print(f"Model: {model.name}")
                print(f"Downloads: {model.downloads:,}")
                print(f"Likes: {model.likes:,}")
                print(f"Pipeline: {model.pipeline_tag}")
                print(f"Tags: {', '.join(model.tags[:5])}")
        else:
            # Scrape trending
            print("Scraping trending models...")
            models = await scrape_top_models(100)
            print(f"Got {len(models)} models")
            
            # Save to file
            with open("data/trending_models.json", "w") as f:
                json.dump(models, f, indent=2)
            print("Saved to data/trending_models.json")
    
    asyncio.run(main())
