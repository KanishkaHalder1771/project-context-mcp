"""
Configuration management for Project Context MCP Server
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import Optional


class AzureConfig(BaseModel):
    """Azure OpenAI configuration for LLM services"""
    endpoint: str
    api_key: str
    deployment_name: str
    api_version: str = "2025-01-01-preview"


class Settings(BaseSettings):
    """Configuration settings for the MCP server"""
    
        # Data directories
    data_dir: Path = Path("./data")
    vector_store_path: Optional[Path] = None
    graph_file: Optional[Path] = None
    metadata_db: Optional[Path] = None
        
        # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
        
        # Similarity thresholds
    duplicate_threshold: float = 0.8
    similarity_threshold: float = 0.6
        
        # Auto-categorization
    category_confidence_threshold: float = 0.7
    
    # Logging
    log_level: str = "INFO"
    
    # Neo4j Graph Database Configuration (read from .env)
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    
    # Azure OpenAI Configuration (read from .env)
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_api_version: str = "2025-01-01-preview"
    azure_openai_deployment_name: Optional[str] = None
    azure_openai_embedding_deployment: Optional[str] = None
    
    # Azure OpenAI Embedding Configuration (separate endpoint)
    azure_openai_embedding_endpoint: Optional[str] = None
    azure_openai_embedding_api_key: Optional[str] = None
    azure_openai_embedding_api_version: str = "2023-05-15"
    
    # Azure SLM (Small Language Model) Configuration (read from .env)
    azure_slm_endpoint: Optional[str] = None
    azure_slm_api_key: Optional[str] = None
    azure_slm_api_version: str = "2025-01-01-preview"
    azure_slm_deployment_name: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set up dependent paths after initialization
        if self.vector_store_path is None:
            self.vector_store_path = self.data_dir / "vector_store"
        if self.graph_file is None:
            self.graph_file = self.data_dir / "knowledge_graph.pkl"
        if self.metadata_db is None:
            self.metadata_db = self.data_dir / "contexts.db"
    
    @property
    def azure_config(self) -> AzureConfig:
        """Get validated Azure OpenAI configuration for main LLM"""
        if not all([self.azure_openai_endpoint, self.azure_openai_api_key, self.azure_openai_deployment_name]):
            raise ValueError("Azure OpenAI configuration incomplete. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME in .env")
        
        return AzureConfig(
            endpoint=self.azure_openai_endpoint,  # type: ignore
            api_key=self.azure_openai_api_key,  # type: ignore
            deployment_name=self.azure_openai_deployment_name,  # type: ignore
            api_version=self.azure_openai_api_version
        )
    
    @property
    def azure_embedding_config(self) -> AzureConfig:
        """Get validated Azure OpenAI configuration for embeddings"""
        # Use embedding-specific config or fallback to main config
        endpoint = self.azure_openai_embedding_endpoint or self.azure_openai_endpoint
        api_key = self.azure_openai_embedding_api_key or self.azure_openai_api_key
        deployment = self.azure_openai_embedding_deployment or "text-embedding-ada-002"
        api_version = self.azure_openai_embedding_api_version or self.azure_openai_api_version
        
        if not all([endpoint, api_key]):
            raise ValueError("Azure OpenAI embedding configuration incomplete")
        
        return AzureConfig(
            endpoint=endpoint,  # type: ignore
            api_key=api_key,  # type: ignore
            deployment_name=deployment,
            api_version=api_version
        )
    
    @property
    def azure_slm_config(self) -> Optional[AzureConfig]:
        """Get validated Azure SLM configuration (optional)"""
        if not all([self.azure_slm_endpoint, self.azure_slm_api_key, self.azure_slm_deployment_name]):
            return None
        
        return AzureConfig(
            endpoint=self.azure_slm_endpoint,  # type: ignore
            api_key=self.azure_slm_api_key,  # type: ignore
            deployment_name=self.azure_slm_deployment_name,  # type: ignore
            api_version=self.azure_slm_api_version
        )


# Global settings instance
settings = Settings() 