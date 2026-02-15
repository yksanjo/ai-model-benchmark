# AI Model Benchmark Scraper

A tool that scrapes Hugging Face model pages, extracts benchmark performance data, and helps detect overclaiming in AI model reports.

## Why This Exists

AI companies often report benchmark results under optimal conditions that don't reflect real-world usage. This tool:
- Scrapes public benchmark data from Hugging Face
- Compares reported vs. actual performance
- Identifies cherry-picked or inflated metrics
- Provides transparency for ML engineers and buyers

## Features

- 🤖 Scrape model pages from Hugging Face
- 📊 Extract benchmark metrics (MMLU, HumanEval, etc.)
- 🔍 Detect inconsistent or cherry-picked results
- 📈 Track performance drift over time
- 📝 Generate comparison reports

## Quick Start

```bash
# Clone and install
git clone https://github.com/yksanjo/ai-model-benchmark.git
cd ai-model-benchmark
pip install -r requirements.txt

# Run the scraper
python main.py --model meta-llama/Llama-2-7b

# Or run benchmarks
python run_benchmark.py --model meta-llama/Llama-2-7b --task mmlu
```

## Project Structure

```
ai-model-benchmark/
├── scraper/           # HF page scraping logic
├── extractors/        # Benchmark data extraction
├── benchmarks/        # Actual benchmark running
├── reports/           # Report generation
├── data/             #存储已抓取的数据
└── main.py           # Entry point
```

## Usage Examples

```python
from scraper.hf_scraper import HuggingFaceScraper

scraper = HuggingFaceScraper()
model_data = scraper.get_model_info("meta-llama/Llama-2-7b")
print(model_data.benchmarks)
```

## Tech Stack

- Python 3.10+
- httpx - Async HTTP client
- BeautifulSoup - HTML parsing
- Pandas - Data handling
- SQLite - Local database

## Roadmap

- [x] Basic HF scraper
- [ ] Extract benchmark metrics
- [ ] Run actual benchmarks
- [ ] Compare reported vs actual
- [ ] Generate PDF reports
- [ ] Web dashboard

## License

MIT
