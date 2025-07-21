"""
Neo4j Knowledge Graph Pipeline Components
"""

from typing import Optional, Union, Any, Sequence, List
import neo4j
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.components.resolver import BasePropertySimilarityResolver
from neo4j_graphrag.experimental.pipeline.config.template_pipeline.simple_kg_builder import SimpleKGPipelineConfig
from neo4j_graphrag.experimental.components.types import ResolutionStats
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


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
    
    async def run(self) -> ResolutionStats:
        return await super().run()
    
    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        Use LLM to compute similarity between two entity texts
        Returns a similarity score between 0.0 and 1.0
        """
        if not self.llm:
            # Fallback to simple string matching if no LLM available
            return 1.0 if text_a.lower() == text_b.lower() else 0.0
        
        # LLM prompt for similarity scoring
        prompt = f"""I have a neo4j graph with some software services and technologies as nodes, i want to merge any similar or duplicate nodes, so for that i need you to tell me if the following nodes are the same or not. 
Node1: {text_a}
Node2: {text_b}
give me just a number between 0 to 1, 1 is same, 0 is not same, it will act as the scoring of how similar these nodes are. just give this number as output no explanation or anything, just the number"""
        
        try:
            # Call the small LLM to get similarity score
            response = self.llm.invoke(prompt)
            # Extract the numeric score from response
            score_text = response.content.strip()
            
            # Make API call to local vLLM server
            import requests
            
            api_url = "http://localhost:8000/v1/chat/completions"
            payload = {
                "model": "Qwen/Qwen3-1.7B",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"I have a neo4j graph with some software services and technologies as nodes, i want to merge any similar or duplicate nodes, so for that i need you to tell me if the following nodes are the same or not. Node1: {text_a} Node2: {text_b}, give me just a number between 0 to 1, 1 is same, 0 is not same, it will act as the scoring of how similar these nodes are. just give this number as output no explanation or anything, just the number"}
                ],
                "max_tokens": 2000
            }
            
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
            response_data = response.json()
            score_text = response_data['choices'][0]['message']['content'].strip()
            
            print(f"Comparing Components: {text_a} and {text_b}")
            print(f"Score text: {score_text}")
            print(f"Response: {response_data}")
            # Try to parse the score as a float
            try:
                similarity_score = float(score_text)
                # Clamp the score between 0.0 and 1.0
                return min(max(similarity_score, 0.0), 1.0)
            except ValueError:
                # If parsing fails, try to extract first number from response
                print(f"⚠️ Could not parse similarity score from: {score_text}")
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

    entity_resolution_type: str = "llm"
    resolution_similarity_threshold: float = 0.8
    slm: Optional[LLMInterface] = None
    
    def __init__(self, entity_resolution_type="exact", resolution_similarity_threshold=0.8, slm: Optional[LLMInterface] = None, **kwargs):
        super().__init__(**kwargs)
        self.entity_resolution_type = entity_resolution_type
        self.resolution_similarity_threshold = resolution_similarity_threshold
        self.slm = slm
    
    def _get_resolver(self):
        """Override to use different resolvers based on entity_resolution_type"""
        print(f"Getting resolver for entity resolution type: {self.entity_resolution_type}")
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
            from neo4j_graphrag.experimental.components.resolver import SpaCySemanticMatchResolver
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
            # Use the provided SLM for entity resolution
            if not self.slm:
                print("⚠️ SLM not provided for LLM resolver")
                return None
            
            return LLMSimilarityResolver(
                driver=self.get_default_neo4j_driver(),
                neo4j_database=self.neo4j_database,
                similarity_threshold=self.resolution_similarity_threshold,
                resolve_properties=["name", "description"],
                llm=self.slm
            )
        else:
            raise ValueError(f"Unknown entity resolution type: {self.entity_resolution_type}")


class CustomKGPipeline:
    """
    A custom knowledge graph pipeline that mirrors SimpleKGPipeline but uses CustomKGPipelineConfig
    for enhanced entity resolution capabilities including LLM-based resolution.

    Args:
        llm (LLMInterface): An instance of an LLM to use for entity and relation extraction.
        driver (neo4j.Driver): A Neo4j driver instance for database connection.
        embedder (Embedder): An instance of an embedder used to generate chunk embeddings from text chunks.
        slm (Optional[LLMInterface]): An optional small language model for entity resolution.
        entity_resolution_type (str): Type of entity resolution ("exact", "spacy", "fuzzy", "llm", "none").
        resolution_similarity_threshold (float): Similarity threshold for entity resolution.
        entities (Optional[Sequence[EntityInputType]]): DEPRECATED. Use schema instead.
        relations (Optional[Sequence[RelationInputType]]): DEPRECATED. Use schema instead.
        potential_schema (Optional[List[tuple]]): DEPRECATED. Use schema instead.
        schema (Optional[Union[GraphSchema, dict[str, list]]]): Schema configuration for the knowledge graph.
        from_pdf (bool): Whether to process PDF files or text directly.
        text_splitter (Optional[TextSplitter]): A text splitter component.
        pdf_loader (Optional[DataLoader]): A PDF loader component.
        kg_writer (Optional[KGWriter]): A knowledge graph writer component.
        on_error (str): Error handling strategy.
        prompt_template (Union[ERExtractionTemplate, str]): Custom prompt template for extraction.
        perform_entity_resolution (bool): Whether to perform entity resolution.
        lexical_graph_config (Optional[LexicalGraphConfig]): Lexical graph configuration.
        neo4j_database (Optional[str]): Neo4j database name.
    """

    def __init__(
        self,
        llm: LLMInterface,
        driver: neo4j.Driver,
        embedder: Embedder,
        slm: Optional[LLMInterface] = None,
        entity_resolution_type: str = "exact",
        resolution_similarity_threshold: float = 0.8,
        entities: Optional[Any] = None,
        relations: Optional[Any] = None,
        potential_schema: Optional[Any] = None,
        schema: Optional[Union[dict, Any]] = None,
        from_pdf: bool = True,
        text_splitter: Optional[Any] = None,
        pdf_loader: Optional[Any] = None,
        kg_writer: Optional[Any] = None,
        on_error: str = "IGNORE",
        prompt_template: Optional[str] = None,
        perform_entity_resolution: bool = True,
        lexical_graph_config: Optional[Any] = None,
        neo4j_database: Optional[str] = None,
    ):
        try:
            # Import required classes
            from neo4j_graphrag.experimental.pipeline.config.runner import PipelineRunner
            from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
            from neo4j_graphrag.experimental.pipeline.config.object_config import ComponentType
            from neo4j_graphrag.experimental.components.entity_relation_extractor import OnError
            from neo4j_graphrag.generation.prompts import ERExtractionTemplate
            from pydantic import ValidationError
            
            # Create our custom config exactly like SimpleKGPipeline does
            config = CustomKGPipelineConfig.model_validate(
                dict(
                    llm_config=llm,
                    neo4j_config=driver,
                    embedder_config=embedder,
                    slm=slm,
                    entity_resolution_type=entity_resolution_type,
                    resolution_similarity_threshold=resolution_similarity_threshold,
                    entities=entities or [],
                    relations=relations or [],
                    potential_schema=potential_schema,
                    schema=schema,
                    from_pdf=from_pdf,
                    pdf_loader=ComponentType(pdf_loader) if pdf_loader else None,
                    kg_writer=ComponentType(kg_writer) if kg_writer else None,
                    text_splitter=ComponentType(text_splitter) if text_splitter else None,
                    on_error=OnError(on_error),
                    prompt_template=prompt_template or ERExtractionTemplate(),
                    perform_entity_resolution=perform_entity_resolution,
                    lexical_graph_config=lexical_graph_config,
                    neo4j_database=neo4j_database,
                )
            )
        except (ValidationError, ValueError) as e:
            from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
            raise PipelineDefinitionError() from e

        # Create the runner from our custom config (exactly like SimpleKGPipeline)
        from neo4j_graphrag.experimental.pipeline.config.runner import PipelineRunner
        self.runner = PipelineRunner.from_config(config)

    async def run_async(
        self, file_path: Optional[str] = None, text: Optional[str] = None
    ):
        """
        Asynchronously runs the knowledge graph building process.

        Args:
            file_path (Optional[str]): The path to the PDF file to process. Required if `from_pdf` is True.
            text (Optional[str]): The text content to process. Required if `from_pdf` is False.

        Returns:
            The result of the pipeline execution.
        """
        return await self.runner.run({"file_path": file_path, "text": text})
    
    def run_sync(self, file_path: Optional[str] = None, text: Optional[str] = None):
        """
        Synchronously runs the knowledge graph building process.

        Args:
            file_path (Optional[str]): The path to the PDF file to process. Required if `from_pdf` is True.
            text (Optional[str]): The text content to process. Required if `from_pdf` is False.

        Returns:
            The result of the pipeline execution.
        """
        import asyncio
        return asyncio.run(self.run_async(file_path=file_path, text=text))
