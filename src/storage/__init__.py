"""
Storage module - coordinates vector DB, graph DB, and metadata
"""

# from .vector_db import VectorDB
# from .graph_db import GraphDB
# import sqlite3


class StorageManager:
    """Coordinates all storage systems"""
    
    def __init__(self, settings):
        self.settings = settings
        # TODO: Initialize storage components
        # self.vector_db = VectorDB(settings)
        # self.graph_db = GraphDB(settings)
        # self.metadata_conn = None
    
    async def initialize(self):
        """Initialize all storage systems"""
        # TODO: Implement initialization
        pass
    
    def save_metadata(self, context_id: str, title: str, category: str):
        """Save context metadata to SQLite"""
        # TODO: Implement metadata storage
        pass
    
    def get_categories(self):
        """Get all existing categories with counts"""
        # TODO: Implement category retrieval
        return [] 