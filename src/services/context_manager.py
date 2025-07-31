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
    
    async def store_context(self, content: str) -> Dict[str, Any]:
        """
        Store context using GraphBuilder - simple implementation
        
        Args:
            content: The context content to store
            
        Returns:
            Simple dict with success status and message
        """
        try:
            print(f"🔍 ContextManager: Storing context (length: {len(content)} chars)")
            
            # Use GraphBuilder to process and store the content
            self._graph_builder.build_graph_from_text_sync(content)
            print("✅ ContextManager: Content successfully processed by GraphBuilder")
            
            return {
                'success': True,
                'message': 'Context stored successfully'
            }
            
        except Exception as e:
            print(f"❌ ContextManager: Failed to store context: {str(e)}")
            return {
                'success': False,
                'message': 'Failed to insert context'
            } 