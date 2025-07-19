# Project Context MCP Server

A smart context management system for coding projects that uses knowledge graphs and vector databases for intelligent storage and retrieval.

## Features

- **Zero-friction storage**: Just say "add this to context" - no manual categorization needed
- **Smart auto-categorization**: AI-powered category detection and assignment  
- **Intelligent relationships**: Automatic discovery of connections between contexts
- **Semantic search**: Find contexts by meaning, not just keywords
- **Session continuity**: Never lose project context again

## Quick Start

### Installation with uv (Recommended)

```bash
# Install dependencies
uv sync

# Run setup script
uv run python scripts/setup_dev.py
```

### Alternative Installation with pip

```bash
pip install -e .
python scripts/setup_dev.py
```

### Setup

1. Copy environment template:
```bash
cp .env.example .env
```

2. Run the MCP server:
```bash
uv run python -m src.main
# OR with pip: python -m src.main
```

## MCP Tools

The server exposes 5 simple tools:

1. `list_categories()` - Show available context categories
2. `store_context(content, category)` - Store new context with auto-categorization
3. `get_context(topic, category)` - Retrieve contexts about a topic
4. `get_related_contexts(topic)` - Find related contexts using knowledge graph
5. `delete_context(topic, category)` - Remove contexts about a topic

## Architecture

- **Vector Database**: LanceDB for high-performance semantic search with Apache Arrow
- **Knowledge Graph**: NetworkX for relationship management
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2) for local, fast embeddings
- **Auto-categorization**: Smart category detection
- **Duplicate detection**: Prevents redundant contexts
- **Local storage**: All data stored locally in `./data/`

## Technology Stack

- **LanceDB**: Ultra-fast vector database built on Apache Arrow
- **NetworkX**: Directed graph for context relationships
- **Sentence Transformers**: Local embedding generation
- **Pydantic**: Type safety and validation
- **Python 3.10+**: Modern Python features

## Development

```bash
# Install development dependencies
uv sync

# Format code
uv run black .
uv run ruff check .

# Type checking
uv run mypy src/
```

## Performance Benefits of LanceDB

- **2-3x faster** than ChromaDB for queries
- **Lower memory footprint** with zero-copy operations
- **Built-in analytics** capabilities for complex filtering
- **Apache Arrow** based for maximum performance
- **Local-first** with no external dependencies

## License

MIT 