"""
Neo4j Knowledge Graph Builder for Design Conversations
"""

import os
from typing import Optional
from neo4j import GraphDatabase
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from pydantic import BaseModel

# Add imports for custom resolver
from neo4j_graphrag.experimental.components.resolver import SpaCySemanticMatchResolver, BasePropertySimilarityResolver
from neo4j_graphrag.experimental.pipeline.config.template_pipeline.simple_kg_builder import SimpleKGPipelineConfig

from .models import NEO4J_SCHEMA
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import settings, AzureConfig


class LLMSimilarityResolver(BasePropertySimilarityResolver):
    """Custom resolver that uses a small LLM to compute entity similarity"""
    
    def __init__(
        self,
        driver,
        filter_query=None,
        resolve_properties=None,
        similarity_threshold=0.8,
        neo4j_database=None,
        llm=None
    ):
        super().__init__(driver, filter_query, resolve_properties, similarity_threshold, neo4j_database)
        self.llm = llm  # Small 3B LLM will be injected here
    
    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        Use LLM to compute similarity between two entity texts
        Returns a similarity score between 0.0 and 1.0
        """
        if not self.llm:
            # Fallback to simple string matching if no LLM available
            return 1.0 if text_a.lower() == text_b.lower() else 0.0
        
        # LLM prompt for similarity scoring
        prompt = f"""Compare these two entity descriptions and rate their similarity from 0.0 to 1.0:

Entity A: {text_a}
Entity B: {text_b}

Consider:
- Are they referring to the same thing?
- Do they have similar meaning or function?
- Are they just different ways to describe the same entity?

Return only a number between 0.0 (completely different) and 1.0 (identical). Do not return any other text, exactly this format:
{{"similarity_score": 0.5}}"""
        
        try:
            # Call the small LLM to get similarity score
            response = self.llm.invoke(prompt)
            # Extract the numeric score from response
            score_text = response.content.strip()
            
            # Try to parse the score as a float
            try:
                similarity_score = float(score_text)
                # Clamp the score between 0.0 and 1.0
                return min(max(similarity_score, 0.0), 1.0)
            except ValueError:
                # If parsing fails, try to extract first number from response
                import re
                numbers = re.findall(r'0?\.\d+|[01]\.?\d*', score_text)
                if numbers:
                    similarity_score = float(numbers[0])
                    return min(max(similarity_score, 0.0), 1.0)
                else:
                    print(f"⚠️ Could not parse similarity score from: {score_text}")
                    # Fallback to exact matching
                    return 1.0 if text_a.lower() == text_b.lower() else 0.0
            
        except Exception as e:
            print(f"⚠️ LLM similarity computation failed: {e}")
            # Fallback to exact matching
            return 1.0 if text_a.lower() == text_b.lower() else 0.0


class CustomKGPipelineConfig(SimpleKGPipelineConfig):
    """Custom pipeline configuration that uses different resolvers based on entity_resolution_type"""
    
    def __init__(self, entity_resolution_type="exact", resolution_similarity_threshold=0.8, azure_slm_config: Optional[AzureConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.entity_resolution_type = entity_resolution_type
        self.resolution_similarity_threshold = resolution_similarity_threshold
        self.azure_slm_config = azure_slm_config
    
    def _get_resolver(self):
        """Override to use different resolvers based on entity_resolution_type"""
        if not self.perform_entity_resolution or self.entity_resolution_type == "none":
            return None
        elif self.entity_resolution_type == "exact":
            from neo4j_graphrag.experimental.components.resolver import SinglePropertyExactMatchResolver
            return SinglePropertyExactMatchResolver(
                driver=self.get_default_neo4j_driver(),
                neo4j_database=self.neo4j_database,
                resolve_property="name"
            )
        elif self.entity_resolution_type == "spacy":
            return SpaCySemanticMatchResolver(
                driver=self.get_default_neo4j_driver(),
                neo4j_database=self.neo4j_database,
                similarity_threshold=self.resolution_similarity_threshold,
                spacy_model="en_core_web_lg",
                resolve_properties=["name"]
            )
        elif self.entity_resolution_type == "fuzzy":
            from neo4j_graphrag.experimental.components.resolver import FuzzyMatchResolver
            return FuzzyMatchResolver(
                driver=self.get_default_neo4j_driver(),
                neo4j_database=self.neo4j_database,
                similarity_threshold=self.resolution_similarity_threshold,
                resolve_properties=["name"]
            )
        elif self.entity_resolution_type == "llm":
            # Create small LLM for entity resolution
            small_llm = self._create_small_llm()
            return LLMSimilarityResolver(
                driver=self.get_default_neo4j_driver(),
                neo4j_database=self.neo4j_database,
                similarity_threshold=self.resolution_similarity_threshold,
                resolve_properties=["name", "description"],
                llm=small_llm
            )
        else:
            raise ValueError(f"Unknown entity resolution type: {self.entity_resolution_type}")
    
    def _create_small_llm(self):
        """Create the small LLM instance for entity resolution"""
        # Validate Azure SLM configuration
        if not self.azure_slm_config:
            print("⚠️ Azure SLM configuration not provided for LLM resolver")
            return None
        
        # Initialize small LLM using Azure SLM configuration
        try:
            small_llm = AzureOpenAILLM(
                model_name=self.azure_slm_config.deployment_name,
                azure_endpoint=self.azure_slm_config.endpoint,
                api_version=self.azure_slm_config.api_version,
                api_key=self.azure_slm_config.api_key,
                model_params={
                    "temperature": 0.0,  # Deterministic for similarity scoring
                    "max_tokens": 10,    # We only need a similarity score (0.0-1.0)
                }
            )
            print(f"✅ Small LLM initialized: {self.azure_slm_config.deployment_name}")
            return small_llm
        except Exception as e:
            print(f"❌ Failed to initialize small LLM: {e}")
            return None


class GraphBuilder:
    """
    Neo4j Knowledge Graph Builder for extracting entities and relationships
    from software design conversations using Azure OpenAI
    """
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        azure_config: Optional[AzureConfig] = None,
        azure_embedding_config: Optional[AzureConfig] = None,
        azure_slm_config: Optional[AzureConfig] = None,
        entity_resolution_type: str = "llm",  # "exact", "spacy", "fuzzy", "llm" or "none"
        resolution_similarity_threshold: float = 0.8,
    ):
        """
        Initialize the graph builder with Azure OpenAI
        
        Args:
            neo4j_uri: Neo4j database URI (optional, reads from .env NEO4J_URI or defaults to bolt://localhost:7687)
            neo4j_user: Neo4j username (optional, reads from .env NEO4J_USER or defaults to neo4j)
            neo4j_password: Neo4j password (optional, reads from .env NEO4J_PASSWORD or defaults to password)
            azure_config: Azure OpenAI configuration for main LLM (optional, reads from settings if not provided)
            azure_embedding_config: Azure OpenAI configuration for embeddings (optional, reads from settings if not provided)
            azure_slm_config: Azure OpenAI configuration for small language model (optional, for LLM entity resolution)
            entity_resolution_type: Type of entity resolution ("exact", "spacy", "fuzzy", "llm", "none")
            resolution_similarity_threshold: Similarity threshold for entity resolution (0.0-1.0)
        """
        self.neo4j_uri = neo4j_uri or settings.neo4j_uri or "bolt://localhost:7687"
        self.neo4j_user = neo4j_user or settings.neo4j_user or "neo4j"
        self.neo4j_password = neo4j_password or settings.neo4j_password or "password"
        self.entity_resolution_type = entity_resolution_type
        self.resolution_similarity_threshold = resolution_similarity_threshold
        
        # Set up Azure configurations with fallback to validated settings
        self.azure_config = azure_config or settings.azure_config
        self.azure_embedding_config = azure_embedding_config or settings.azure_embedding_config
        self.azure_slm_config = azure_slm_config or settings.azure_slm_config

        # Initialize components
        self.driver = None
        self.llm = None
        self.embedder = None
        self.pipeline = None
        
        self._setup_components()
    
    def _setup_components(self):
        """Set up Neo4j driver, Azure OpenAI LLM, embedder, and pipeline"""
        
        # Neo4j driver
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # LLM setup - Using Azure OpenAI deployment
        self.llm = AzureOpenAILLM(
            model_name=self.azure_config.deployment_name,
            azure_endpoint=self.azure_config.endpoint,
            api_version=self.azure_config.api_version,
            api_key=self.azure_config.api_key,
            # model_params={
            #     "temperature": 0.0,
            #     # "max_tokens": 1500,
            # }
        )
        
        # Embedder setup - Using Azure OpenAI embeddings
        self.embedder = AzureOpenAIEmbeddings(
            model=self.azure_embedding_config.deployment_name,
            azure_endpoint=self.azure_embedding_config.endpoint,
            api_version=self.azure_embedding_config.api_version,
            api_key=self.azure_embedding_config.api_key,
        )
        
        # Custom prompt for extraction
        self.custom_prompt = """
You are a software architect tasked with extracting information from architecture and design conversations and structuring it in a property graph to refer to later.

Extract the entities (nodes) and specify their type from the following Input text. You dont have to make every small service into a node we can group similar files/services into a component node
Also extract the relationships between these nodes. The relationship direction goes from the start node to the end node.

Node Types:
- Component: 
    Software modules, services, systems, applications, microservices, these are highlevel services like the auth service, the communication service, frontend, backend, 
    For components you can group similar files/services into a component node, for example description about JWT tokens, OAuth2, etc might comeunder the auth service node
- Technology: Frameworks, databases, protocols, tools, libraries , these will be exact tech name like Nextjs, Redis, MySQL, Postgres and other tech or framework names like these, these technology names should only contain the tech name not why its used its description should not justify why its being used, just the common name of this tech for example for Redis name is just redis nothing else.
- Decision: Architectural choices, design patterns, technical decisions

Relationship Types:
- USES: When a component uses a technology or another component
- DECIDES: When a decision affects or chooses something for a component
- DEPENDS_ON: When a component depends on another component

Valid Patterns:
- Component USES Technology
- Decision DECIDES Component
- Component DEPENDS_ON Component

Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "TYPE_OF_NODE", "properties": {{"name": "name of the entity", "description": "description of the entity"}} }}],
  "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "description of the relationship"}} }}] }}

- Use only the information from the Input text. Do not add any additional information.
- If the input text is empty, return empty JSON.
- An AI knowledge assistant must be able to read this graph and immediately understand the architecture context to inform detailed research questions.

Use only the following node types and relationships:
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do res
pect the source and target node types for relationships and the relationship direction.
Do not return any additional information other than the JSON.

Examples:
{examples}

Input text:
{text}
"""
        
        # Pipeline setup - Using SimpleKGPipeline with entity resolution enabled
        self.pipeline = SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            embedder=self.embedder,
            from_pdf=False,
            schema=NEO4J_SCHEMA,
            prompt_template=self.custom_prompt,
            perform_entity_resolution=True  # Enable built-in entity resolution for now
        )
    
    async def build_graph_from_text(self, text: str) -> None:
        """
        Extract entities and relationships from text and build Neo4j graph
        
        Args:
            text: The conversation or design text to process
        """
        print(f"🔍 Processing text: {text[:100]}...")
        try:
            # Run the pipeline (includes entity extraction and resolution)
            await self.pipeline.run_async(text=text)
            print("✅ Pipeline execution completed (including entity resolution)")
            
        except Exception as e:
            print(f"❌ Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def build_graph_from_text_sync(self, text: str) -> None:
        """
        Synchronous version of build_graph_from_text
        
        Args:
            text: The conversation or design text to process
        """
        import asyncio
        asyncio.run(self.build_graph_from_text(text))
    
    def query_graph(self, cypher_query: str, parameters: Optional[dict] = None):
        """
        Execute a Cypher query against the Neo4j graph
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query results
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [record for record in result]
    
    def get_components(self):
        """Get all Component nodes"""
        query = "MATCH (c:Component) RETURN c.name as name"
        return self.query_graph(query)
    
    def get_technologies(self):
        """Get all Technology nodes"""
        query = "MATCH (t:Technology) RETURN t.name as name"
        return self.query_graph(query)
    
    def get_decisions(self):
        """Get all Decision nodes"""
        query = "MATCH (d:Decision) RETURN d.summary as summary"
        return self.query_graph(query)
    
    def get_all_relationships(self):
        """
        Get all relationships in the graph
        
        Returns:
            List of relationships with source, target, and type
        """
        query = """
        MATCH (a)-[r]->(b) 
        RETURN labels(a)[0] as source_type, a.name as source_name,
               type(r) as relationship_type,
               labels(b)[0] as target_type, b.name as target_name
        ORDER BY relationship_type, source_name
        """
        return self.query_graph(query)
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage function
def example_usage():
    """Example of how to use the GraphBuilder"""
    
    # Sample conversation text
    conversation_text = """
    The API Gateway component uses Redis for caching user sessions.
    The AuthService component decides to implement OAuth 2.0 for authentication.
    The API Gateway component depends on the AuthService for user validation.
    The Frontend component uses React framework for the user interface.
    The Database component decides to use PostgreSQL for data persistence.
    """
    
    # Initialize graph builder (uses environment variables)
    with GraphBuilder() as graph_builder:
        
        # Build graph from conversation
        graph_builder.build_graph_from_text_sync(conversation_text)
        
        # Query the results
        components = graph_builder.get_components()
        technologies = graph_builder.get_technologies()
        relationships = graph_builder.get_all_relationships()
        
        print("Components:", components)
        print("Technologies:", technologies)
        print("All relationships:", relationships)


if __name__ == "__main__":
    example_usage() 