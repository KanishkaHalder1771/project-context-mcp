"""
Neo4j Knowledge Graph Builder for Design Conversations
"""
from typing import Optional
from neo4j import GraphDatabase
from neo4j_graphrag.llm import AzureOpenAILLM, OpenAILLM
from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings


# Add imports for custom resolver
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from .models import NEO4J_SCHEMA
from .pipeline import CustomKGPipeline
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import settings, AzureConfig





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
- Decision: Architectural choices, design patterns, technical decisions, these will be major tech decisions like using REST or gRPC for APIs, or using NoSQL or SQL database, not simple decisions to just use a specific tech or framework, if a detailed information is given why a specific tech is being selected then it should be a decision node.

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
        
        # Create SLM if available for LLM-based entity resolution
        slm = None
        if self.azure_slm_config and self.entity_resolution_type == "llm":
            try:
                # slm = AzureOpenAILLM(
                #     model_name=self.azure_slm_config.deployment_name,
                #     azure_endpoint=self.azure_slm_config.endpoint,
                #     api_version=self.azure_slm_config.api_version,
                #     api_key=self.azure_slm_config.api_key,
                #     model_params={
                #         "temperature": 0.8,  # Deterministic for similarity scoring
                #         "max_tokens": 2048,    # We only need a similarity score (0.0-1.0)
                #         "top_p": 0.1,
                #         "frequency_penalty": 0.0,
                #         "presence_penalty": 0.0
                #     }
                # )
                slm = OpenAILLM(
                    model_name="Qwen/Qwen3-1.7B",
                    api_key=self.azure_slm_config.api_key,
                    base_url="http://localhost:8000/v1"
                )
                print(f"✅ Small LLM initialized: {self.azure_slm_config.deployment_name}")
            except Exception as e:
                print(f"❌ Failed to initialize small LLM: {e}")
                slm = None

        # Pipeline setup - Using CustomKGPipeline with configurable entity resolution
        self.pipeline = CustomKGPipeline(
            llm=self.llm,
            driver=self.driver,
            embedder=self.embedder,
            slm=slm,
            entity_resolution_type=self.entity_resolution_type,
            resolution_similarity_threshold=self.resolution_similarity_threshold,
            schema=NEO4J_SCHEMA,
            prompt_template=self.custom_prompt,
            perform_entity_resolution=True,
            from_pdf=False,
            neo4j_database=None,
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