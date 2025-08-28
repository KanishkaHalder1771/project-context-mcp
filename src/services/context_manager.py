"""
Context Manager Service

Handles all context storage, retrieval, and management operations
using a singleton GraphBuilder instance.
"""

from typing import Dict, Any

# Import GraphBuilder from storage.graph
from ..storage.graph import GraphBuilder


class ContextManager:
    """Manages all context operations using GraphBuilder"""
    
    _instance = None
    _graph_builder = None
    
    def __new__(cls):
        """Singleton pattern for ContextManager"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the context manager with singleton GraphBuilder"""
        if self._graph_builder is None:
            # Initialize the singleton GraphBuilder instance
            self._graph_builder = GraphBuilder(
                entity_resolution_type="llm",
                resolution_similarity_threshold=0.7
            )
            print("✅ ContextManager: GraphBuilder singleton initialized")
    
    async def store_context(self, text: str) -> Dict[str, Any]:
        try:
            print(f"🔍 ContextManager: Executing graph query: {text[:100]}...")
            
            # Use GraphBuilder to query the graph
            results = self._graph_builder.build_graph_from_text(text)
            print(f"✅ ContextManager: Query executed successfully, {len(results)} results")
            
            return {
                'success': True,
                'results': [dict(record) for record in results],
                'count': len(results)
            }
            
        except Exception as e:
            print(f"❌ ContextManager: Failed to execute query: {str(e)}")
            return {
                'success': False,
                'message': f'Failed to execute query: {str(e)}',
                'results': [],
                'count': 0
            } 