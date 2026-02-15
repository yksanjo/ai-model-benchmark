#!/usr/bin/env python3
"""
AI Model Benchmark Scraper - CLI Entry Point

A tool to scrape Hugging Face model pages and extract benchmark data.
"""

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from scraper.hf_scraper import HuggingFaceScraper, scrape_model


console = Console()


@click.group()
def cli():
    """AI Model Benchmark Scraper - Compare reported vs actual benchmarks"""
    pass


@cli.command()
@click.argument("model_id")
@click.option("--output", "-o", help="Output file path (JSON)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def scrape(model_id: str, output: str, verbose: bool):
    """Scrape a single model from Hugging Face"""
    
    console.print(f"[bold blue]Scraping model:[/bold blue] {model_id}")
    
    async def run():
        try:
            data = await scrape_model(model_id)
            
            # Display results
            console.print(f"\n[bold green]✓ Successfully scraped![/bold green]")
            console.print(f"Model: {data.metadata.name}")
            console.print(f"Author: {data.metadata.author}")
            console.print(f"Downloads: {data.metadata.downloads:,}")
            console.print(f"Likes: {data.metadata.likes:,}")
            
            if data.pipeline_tag:
                console.print(f"Pipeline: {data.pipeline_tag}")
            
            if data.benchmarks:
                console.print(f"\n[bold]Benchmarks found:[/bold] {len(data.benchmarks)}")
                
                table = Table(show_header=True)
                table.add_column("Benchmark")
                table.add_column("Score")
                table.add_column("Dataset")
                table.add_column("Shots")
                
                for b in data.benchmarks:
                    shots = str(b.num_shots) if b.num_shots else "-"
                    table.add_row(b.name, str(b.value), b.dataset, shots)
                
                console.print(table)
            else:
                console.print("[yellow]No benchmarks found on page[/yellow]")
            
            # Save to file
            output_path = output or f"data/{model_id.replace('/', '_')}.json"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(data.to_dict(), f, indent=2)
            
            console.print(f"\n[dim]Saved to {output_path}[/dim]")
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            if verbose:
                console.print_exception()
            sys.exit(1)
    
    asyncio.run(run())


@cli.command()
@click.argument("model_ids", nargs=-1)
@click.option("--output", "-o", default="data/batch_results.json", help="Output file")
def batch(model_ids: tuple, output: str):
    """Scrape multiple models"""
    
    console.print(f"[bold blue]Scraping {len(model_ids)} models...[/bold blue]")
    
    async def run():
        results = []
        
        async with HuggingFaceScraper() as scraper:
            for model_id in model_ids:
                console.print(f"  Scraping {model_id}...", end=" ")
                try:
                    data = await scraper.get_model_info(model_id)
                    console.print("[green]✓[/green]")
                    results.append(data.to_dict())
                except Exception as e:
                    console.print(f"[red]✗ {e}[/red]")
        
        # Save results
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        
        console.print(f"\n[bold green]Done![/bold green] Saved {len(results)} results to {output}")
    
    asyncio.run(run())


@cli.command()
@click.option("--limit", default=50, help="Number of models to fetch")
@click.option("--output", "-o", default="data/popular_models.json", help="Output file")
def popular(limit: int, output: str):
    """Get popular models list (from HF trending)"""
    
    console.print(f"[bold blue]Fetching top {limit} popular models...[/bold blue]")
    
    async def run():
        async with HuggingFaceScraper() as scraper:
            # Fetch trending page
            html = await scraper.get_model_page("models?sort=downloads")
            # This is a simplified version - HF may change their page structure
            console.print("[yellow]Note: HF trending page parsing not fully implemented[/yellow]")
            console.print("Try using the scrape command with specific model IDs")
    
    asyncio.run(run())


if __name__ == "__main__":
    cli()
