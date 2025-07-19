# Python MCP Server Project Structure

## Project Layout

```
project-context-mcp/
├── pyproject.toml               # Modern Python packaging
├── requirements.txt             # Dependencies
├── README.md                   # Setup and usage docs
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore patterns
│
├── src/
│   └── project_context_mcp/
│       ├── __init__.py
│       ├── main.py             # MCP server entry point
│       ├── config.py           # Configuration management
│       │
│       ├── storage/            # Data persistence layer
│       │   ├── __init__.py
│       │   ├── vector_db.py        # ChromaDB vector database operations
│       │   ├── graph_db.py         # NetworkX knowledge graph operations
│       │   └── models.py           # Data models/schemas (Pydantic)
│       │
│       ├── services/           # Business logic services
│       │   ├── __init__.py
│       │   ├── context_manager.py   # Main context operations orchestrator
│       │   ├── embeddings.py        # Sentence transformers & embedding logic
│       │   ├── categorizer.py       # Auto-categorization service
│       │   ├── similarity.py        # Duplicate detection & semantic search
│       │   └── relationships.py     # Knowledge graph relationship logic
│       │
│       └── tools/              # MCP tool implementations (one file per tool)
│           ├── __init__.py
│           ├── list_categories.py   # Tool 1: list_categories()
│           ├── store_context.py     # Tool 2: store_context()
│           ├── get_context.py       # Tool 3: get_context()
│           ├── get_related.py       # Tool 4: get_related_contexts()
│           └── delete_context.py    # Tool 5: delete_context()
│
├── data/                       # Local data storage (gitignored)
│   ├── contexts.db            # SQLite metadata database
│   ├── vector_store/          # Chroma vector database files
│   └── knowledge_graph.pkl    # NetworkX graph pickle file
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Pytest configuration
│   ├── test_services/        # Test business logic
│   ├── test_tools/           # Test MCP tools
│   └── test_integration/     # End-to-end tests
│
└── scripts/                  # Development utilities
    ├── setup_dev.py         # Development environment setup
    └── demo.py               # Demo script for testing
```

## Key Dependencies (requirements.txt)

```python
# MCP Framework
mcp>=1.0.0

# Vector Database & Embeddings
chromadb>=0.4.0
sentence-transformers>=2.2.0

# Knowledge Graph & Data
networkx>=3.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Utilities
python-dotenv>=1.0.0
loguru>=0.7.0

# Development
pytest>=7.0.0
black>=23.0.0
ruff>=0.1.0

# Optional: Performance boosters
# torch>=2.0.0  # For GPU acceleration of embeddings
# numpy>=1.24.0  # For numerical operations
```

## Simplified Core Components

### 1. **Main Entry Point (main.py)**
```python
#!/usr/bin/env python3
"""
Project Context MCP Server

A smart context management system for coding projects that uses
knowledge graphs and vector databases for intelligent storage and retrieval.
"""

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .config import Settings
from .storage import StorageManager
from .services.context_manager import ContextManager
from .tools import register_all_tools

async def main():
    settings = Settings()
    
    # Initialize storage and services
    storage_manager = StorageManager(settings)
    await storage_manager.initialize()
    
    context_manager = ContextManager(storage_manager, settings)
    
    # Create MCP server and register tools
    server = Server("project-context")
    register_all_tools(server, context_manager)
    
    # Start server
    async with stdio_server() as streams:
        await server.run(streams[0], streams[1])

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. **Storage Manager (storage/__init__.py)**
```python
"""
Storage module - coordinates vector DB, graph DB, and metadata
"""

from .vector_db import VectorDB
from .graph_db import GraphDB
import sqlite3

class StorageManager:
    def __init__(self, settings):
        self.settings = settings
        self.vector_db = VectorDB(settings)
        self.graph_db = GraphDB(settings)
        self.metadata_conn = None
    
    async def initialize(self):
        """Initialize all storage systems"""
        await self.vector_db.initialize()
        await self.graph_db.initialize()
        
        # Metadata DB (SQLite)
        self.metadata_conn = sqlite3.connect(self.settings.metadata_db)
        self._create_tables()
    
    def _create_tables(self):
        """Create SQLite tables for metadata"""
        cursor = self.metadata_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contexts (
                id TEXT PRIMARY KEY,
                title TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.metadata_conn.commit()
    
    # Metadata operations
    def save_metadata(self, context_id: str, title: str, category: str):
        """Save context metadata to SQLite"""
        cursor = self.metadata_conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO contexts (id, title, category, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (context_id, title, category))
        self.metadata_conn.commit()
    
    def get_categories(self):
        """Get all existing categories with counts"""
        cursor = self.metadata_conn.cursor()
        cursor.execute('''
            SELECT category, COUNT(*) as count 
            FROM contexts 
            GROUP BY category 
            ORDER BY count DESC
        ''')
        return cursor.fetchall()
```

### 3. **Vector Database (storage/vector_db.py)**
```python
"""
ChromaDB vector database operations for semantic search
"""

import chromadb
from chromadb.config import Settings as ChromaSettings

class VectorDB:
    def __init__(self, settings):
        self.settings = settings
        self.client = None
        self.collection = None
    
    async def initialize(self):
        """Initialize ChromaDB client and collection"""
        self.settings.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.settings.vector_store_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection("contexts")
    
    async def store(self, context_id: str, content: str, embedding: list, metadata: dict):
        """Store context with embedding in vector DB"""
        self.collection.add(
            ids=[context_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    
    async def search_similar(self, query: str, n_results: int = 10):
        """Semantic search by text query"""
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
    
    async def search_by_embedding(self, embedding: list, n_results: int = 10):
        """Search by embedding vector"""
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
    
    async def get_by_id(self, context_id: str):
        """Get specific context by ID"""
        return self.collection.get(ids=[context_id])
    
    async def update(self, context_id: str, content: str = None, embedding: list = None, metadata: dict = None):
        """Update existing context"""
        update_data = {"ids": [context_id]}
        if content:
            update_data["documents"] = [content]
        if embedding:
            update_data["embeddings"] = [embedding]
        if metadata:
            update_data["metadatas"] = [metadata]
        
        self.collection.update(**update_data)
    
    async def delete(self, context_id: str):
        """Delete context from vector DB"""
        self.collection.delete(ids=[context_id])
    
    def get_collection_info(self):
        """Get collection statistics"""
        return {
            "count": self.collection.count(),
            "name": self.collection.name
        }
```

### 4. **Knowledge Graph (storage/graph_db.py)**
```python
"""
NetworkX knowledge graph operations for relationship management
"""

import networkx as nx
import pickle
from pathlib import Path

class GraphDB:
    def __init__(self, settings):
        self.settings = settings
        self.graph = None
    
    async def initialize(self):
        """Initialize NetworkX graph"""
        self.settings.graph_file.parent.mkdir(parents=True, exist_ok=True)
        self.graph = self._load_graph()
    
    def _load_graph(self):
        """Load existing graph or create new one"""
        if self.settings.graph_file.exists():
            try:
                with open(self.settings.graph_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, FileNotFoundError):
                # If file is corrupted, create new graph
                pass
        return nx.DiGraph()
    
    def save(self):
        """Persist graph to disk"""
        with open(self.settings.graph_file, 'wb') as f:
            pickle.dump(self.graph, f)
    
    def add_node(self, context_id: str, **attributes):
        """Add context node to knowledge graph"""
        self.graph.add_node(context_id, **attributes)
        self.save()
    
    def add_relationship(self, source_id: str, target_id: str, rel_type: str, **attributes):
        """Add relationship edge to knowledge graph"""
        self.graph.add_edge(source_id, target_id, type=rel_type, **attributes)
        self.save()
    
    def find_related(self, context_id: str, max_depth: int = 3):
        """Find related contexts via graph traversal"""
        if context_id not in self.graph:
            return []
        
        related = []
        try:
            paths = nx.single_source_shortest_path(self.graph, context_id, cutoff=max_depth)
            for node_id, path in paths.items():
                if node_id != context_id:
                    # Get edge data for relationship type
                    edge_data = self.graph.get_edge_data(path[-2], node_id) if len(path) > 1 else {}
                    related.append({
                        'context_id': node_id,
                        'distance': len(path) - 1,
                        'path': path,
                        'relationship_type': edge_data.get('type', 'unknown'),
                        'attributes': self.graph.nodes[node_id]
                    })
        except nx.NetworkXError:
            pass
        
        return related
    
    def find_path_between(self, source_id: str, target_id: str):
        """Find shortest path between two contexts"""
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            # Add relationship types along the path
            path_with_relations = []
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                path_with_relations.append({
                    'from': path[i],
                    'to': path[i+1],
                    'relationship': edge_data.get('type', 'unknown')
                })
            return path_with_relations
        except nx.NetworkXNoPath:
            return None
    
    def get_neighbors(self, context_id: str):
        """Get direct neighbors of a context"""
        if context_id not in self.graph:
            return []
        
        neighbors = []
        for neighbor in self.graph.neighbors(context_id):
            edge_data = self.graph.get_edge_data(context_id, neighbor)
            neighbors.append({
                'context_id': neighbor,
                'relationship_type': edge_data.get('type', 'unknown'),
                'attributes': self.graph.nodes[neighbor]
            })
        return neighbors
    
    def remove_node(self, context_id: str):
        """Remove context node and all its edges"""
        if context_id in self.graph:
            self.graph.remove_node(context_id)
            self.save()
    
    def get_graph_stats(self):
        """Get graph statistics"""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "is_connected": nx.is_connected(self.graph.to_undirected()) if self.graph.number_of_nodes() > 0 else False
        }
```

### 5. **Data Models (storage/models.py)**
```python
"""
Pydantic data models for type safety and validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ContextMetadata(BaseModel):
    """Metadata for a stored context"""
    context_id: str
    title: str
    category: str
    created_at: datetime
    updated_at: datetime
    tags: List[str] = Field(default_factory=list)

class ContextContent(BaseModel):
    """Full context with content and metadata"""
    context_id: str
    title: str
    category: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

class RelationshipData(BaseModel):
    """Knowledge graph relationship"""
    source_id: str
    target_id: str
    relationship_type: str
    confidence: Optional[float] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)

class CategoryInfo(BaseModel):
    """Category information with stats"""
    name: str
    count: int
    description: str

class SearchResult(BaseModel):
    """Search result with similarity score"""
    context_id: str
    title: str
    content: str
    category: str
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StorageResult(BaseModel):
    """Result of storing a context"""
    action: str  # "created", "updated", "merged"
    context_id: str
    title: str
    category: str
    relationships_created: List[RelationshipData] = Field(default_factory=list)
    similar_contexts_found: List[str] = Field(default_factory=list)
```

### 6. **Context Manager (services/context_manager.py)**
```python
"""
Main context operations orchestrator
"""

from .embeddings import EmbeddingService
from .categorizer import AutoCategorizer
from .similarity import SimilarityService
from .relationships import RelationshipService
import uuid

class ContextManager:
    def __init__(self, storage_manager, settings):
        self.storage = storage_manager
        self.settings = settings
        
        # Initialize services
        self.embedding_service = EmbeddingService(settings)
        self.categorizer = AutoCategorizer(storage_manager, self.embedding_service)
        self.similarity_service = SimilarityService(storage_manager, self.embedding_service)
        self.relationship_service = RelationshipService(storage_manager)
    
    async def store_context(self, content: str, category: str = "auto"):
        """Universal context storage with smart handling"""
        # Generate unique ID
        context_id = f"ctx_{uuid.uuid4().hex[:8]}"
        
        # Auto-categorize if needed
        if category == "auto":
            category_result = await self.categorizer.detect_category(content)
            category = category_result['category']
        
        # Generate embedding
        embedding = await self.embedding_service.embed_text(content)
        
        # Check for similar contexts
        similar = await self.similarity_service.find_similar(content)
        
        if similar and similar[0]['score'] > self.settings.duplicate_threshold:
            # Handle potential duplicate
            return await self._handle_duplicate(content, similar[0], context_id)
        
        # Generate title from content
        title = self._generate_title(content)
        
        # Store in all systems
        await self._store_new_context(context_id, content, title, category, embedding)
        
        # Create relationships
        relationships = await self.relationship_service.detect_relationships(
            context_id, content, category
        )
        
        return {
            'action': 'created',
            'context_id': context_id,
            'title': title,
            'category': category,
            'relationships': relationships
        }
    
    async def get_contexts_by_topic(self, topic: str, category: str = None, max_results: int = 5):
        """Retrieve contexts about a specific topic"""
        # Use vector DB for semantic search
        results = await self.storage.vector_db.search_similar(topic, n_results=max_results)
        
        if category:
            # Filter by category using metadata
            filtered_results = []
            for i, metadata in enumerate(results['metadatas'][0]):
                if metadata.get('category') == category:
                    filtered_results.append({
                        'context_id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': metadata,
                        'similarity': results['distances'][0][i] if 'distances' in results else 1.0
                    })
            return filtered_results[:max_results]
        
        # Return all results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'context_id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': results['distances'][0][i] if 'distances' in results else 1.0
            })
        
        return formatted_results
    
    async def get_related_contexts(self, topic: str, category: str = None, depth: int = 2):
        """Find related contexts using graph + similarity"""
        # First find contexts matching the topic
        topic_contexts = await self.get_contexts_by_topic(topic, category, max_results=3)
        
        if not topic_contexts:
            return []
        
        # For each found context, get graph-related contexts
        all_related = []
        for ctx in topic_contexts:
            context_id = ctx['context_id']
            related = self.storage.graph_db.find_related(context_id, max_depth=depth)
            all_related.extend(related)
        
        return all_related
    
    async def delete_contexts_by_topic(self, topic: str, category: str = None):
        """Delete contexts matching a topic"""
        # Find matching contexts
        contexts = await self.get_contexts_by_topic(topic, category)
        
        # Delete from all storage systems
        deleted_ids = []
        for ctx in contexts:
            context_id = ctx['context_id']
            await self._delete_context(context_id)
            deleted_ids.append(context_id)
        
        return {'deleted_contexts': deleted_ids}
    
    def get_all_categories(self):
        """Get list of all categories with counts"""
        categories = self.storage.get_categories()
        return [
            {
                'name': cat[0], 
                'count': cat[1],
                'description': self._get_category_description(cat[0])
            } 
            for cat in categories
        ]
    
    # Helper methods
    def _generate_title(self, content: str) -> str:
        """Generate a meaningful title from content"""
        # Simple implementation - take first sentence or 50 chars
        first_line = content.split('\n')[0].strip()
        if len(first_line) > 50:
            return first_line[:47] + "..."
        return first_line or "Untitled Context"
    
    def _get_category_description(self, category: str) -> str:
        """Get description for a category"""
        descriptions = {
            'auth': 'Authentication and authorization',
            'api': 'API endpoints and integrations',
            'database': 'Database schemas and queries',
            'ui_ux': 'User interface components',
            'infrastructure': 'Deployment and infrastructure'
        }
        return descriptions.get(category, 'Custom category')
    
    async def _store_new_context(self, context_id, content, title, category, embedding):
        """Store context in all storage systems"""
        metadata = {'category': category, 'title': title}
        
        # Vector DB
        await self.storage.vector_db.store(context_id, content, embedding, metadata)
        
        # Knowledge Graph
        self.storage.graph_db.add_node(context_id, title=title, category=category)
        
        # Metadata DB
        self.storage.save_metadata(context_id, title, category)
    
    async def _delete_context(self, context_id: str):
        """Delete context from all storage systems"""
        # Vector DB
        await self.storage.vector_db.delete(context_id)
        
        # Knowledge Graph
        self.storage.graph_db.remove_node(context_id)
        
        # Metadata DB
        cursor = self.storage.metadata_conn.cursor()
        cursor.execute("DELETE FROM contexts WHERE id = ?", (context_id,))
        self.storage.metadata_conn.commit()
```

### 4. **MCP Tool Example (tools/store_context.py)**
```python
"""
Store Context MCP Tool
"""

from mcp.server import Server

def register_store_context_tool(server: Server, context_manager):
    
    @server.tool("store_context")
    async def store_context(content: str, category: str = "auto") -> dict:
        """
        Universal context storage tool.
        
        Args:
            content: The context content to store
            category: Target category or "auto" for auto-detection
        """
        
        try:
            result = await context_manager.store_context(content, category)
            
            return {
                "success": True,
                "action": result['action'],
                "title": result['title'], 
                "category": result['category'],
                "context_id": result['context_id'],
                "relationships_created": len(result['relationships'])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### 5. **Tools Registration (tools/__init__.py)**
```python
"""
MCP Tools Registration
"""

from .list_categories import register_list_categories_tool
from .store_context import register_store_context_tool
from .get_context import register_get_context_tool
from .get_related import register_get_related_tool
from .delete_context import register_delete_context_tool

def register_all_tools(server, context_manager):
    """Register all 5 MCP tools"""
    register_list_categories_tool(server, context_manager)
    register_store_context_tool(server, context_manager)
    register_get_context_tool(server, context_manager)
    register_get_related_tool(server, context_manager)
    register_delete_context_tool(server, context_manager)
```

## Configuration (config.py)

```python
"""
Configuration management
"""

from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Data directories
    data_dir: Path = Path("./data")
    vector_store_path: Path = data_dir / "vector_store"
    graph_file: Path = data_dir / "knowledge_graph.pkl"
    metadata_db: Path = data_dir / "contexts.db"
    
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Similarity thresholds
    duplicate_threshold: float = 0.8
    similarity_threshold: float = 0.6
    
    # Auto-categorization
    category_confidence_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
```

## Benefits of Simplified Structure:

1. **Much simpler** - Only 13 files vs 25+ in complex version
2. **Easy to navigate** - One file per tool, services clearly separated
3. **Unified storage** - Single storage.py handles all database connections
4. **Focused services** - Each service has a single responsibility
5. **Clean tools** - One file per MCP tool for easy maintenance
6. **Testable** - Clear interfaces and minimal dependencies
7. **Scalable** - Easy to swap storage backends or add new services

## Quick Start Implementation Order:

1. **storage/models.py** - Define Pydantic models for type safety
2. **storage/vector_db.py** - ChromaDB vector database operations
3. **storage/graph_db.py** - NetworkX knowledge graph operations
4. **storage/__init__.py** - StorageManager that coordinates both databases
5. **services/embeddings.py** - Sentence transformers wrapper
6. **services/context_manager.py** - Main orchestration logic  
7. **tools/store_context.py** - First working MCP tool
8. **tools/list_categories.py** - Simple category listing tool
9. **Remaining tools** - get_context, get_related, delete_context

## Benefits of Modular Storage:

1. **Clear separation** - Vector operations in vector_db.py, graph operations in graph_db.py
2. **Easy testing** - Each storage component can be mocked/tested independently  
3. **Easy swapping** - Replace ChromaDB with Pinecone or NetworkX with Neo4j without touching other code
4. **Type safety** - Pydantic models ensure data consistency across all storage systems
5. **Clean interfaces** - Each storage class has focused, single-purpose methods

This structure makes it **much easier to get started** while being **highly modular** and **production-ready**! 