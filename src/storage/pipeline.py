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
from openai import AzureOpenAI

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
        llm: Optional[AzureOpenAI] = None
    ):
        super().__init__(driver, filter_query, resolve_properties, similarity_threshold, neo4j_database)
        self.llm = llm  # Small 3B LLM will be injected here
    
    async def run(self) -> ResolutionStats:
        """Custom run method that concatenates properties instead of discarding them"""
        from itertools import combinations
        
        match_query = "MATCH (entity:__Entity__)"
        if self.filter_query:
            match_query += f" {self.filter_query}"

        # Get all properties for each entity, grouped by their actual node type
        query = f"""
            {match_query}
            UNWIND labels(entity) AS lab
            WITH lab, entity
            WHERE NOT lab IN ['__Entity__', '__KGBuilder__']
            WITH lab, collect({{
                id: elementId(entity),
                props: properties(entity),
                nodeType: lab
            }}) AS labelCluster
            RETURN lab, labelCluster
        """

        records, _, _ = self.driver.execute_query(query, database_=self.neo4j_database)

        total_entities = 0
        total_merged_nodes = 0

        # for each label/type, process entities
        for row in records:
            node_type = row["lab"]
            entities = row["labelCluster"]

            # Build data for similarity comparison - only using name, grouped by type
            node_names = {}
            node_properties = {}
            
            for ent in entities:
                entity_props = ent["props"]
                node_properties[ent["id"]] = entity_props
                
                # Get name only
                name = str(entity_props.get("name", "")).strip()
                
                if name:  # Only include entities that have a name
                    node_names[ent["id"]] = name
                    
            total_entities += len(node_names)

            # compute pairwise similarity within the same node type and mark those above the threshold
            pairs_to_merge = []
            for (id1, name1), (id2, name2) in combinations(node_names.items(), 2):
                sim = self.compute_similarity(name1, name2, node_type)
                if sim >= self.similarity_threshold:
                    pairs_to_merge.append({id1, id2})

            # consolidate overlapping pairs into unique merge sets
            merged_sets = self._consolidate_sets(pairs_to_merge)

            # perform custom merges with property concatenation
            merged_count = 0
            for node_id_set in merged_sets:
                if len(node_id_set) > 1:
                    # Get all properties from nodes to merge
                    merged_props = {}
                    node_ids = list(node_id_set)
                    
                    for node_id in node_ids:
                        props = node_properties[node_id]
                        for key, value in props.items():
                            if value is not None and str(value).strip():
                                if key == "name":
                                    # For name, keep the first non-empty value
                                    if key not in merged_props:
                                        merged_props[key] = value
                                else:
                                    # For other properties, concatenate with delimiter
                                    if key not in merged_props:
                                        merged_props[key] = str(value)
                                    else:
                                        existing = str(merged_props[key])
                                        new_value = str(value)
                                        if new_value not in existing:  # Avoid duplicates
                                            merged_props[key] = f"{existing} | {new_value}"
                    
                    # Create the merge query with custom property handling
                    merge_query = """
                        MATCH (n) WHERE elementId(n) IN $ids
                        WITH collect(n) AS nodes
                        WITH nodes[0] AS firstNode, nodes
                        SET firstNode += $mergedProps
                        WITH firstNode, nodes[1..] AS otherNodes
                        UNWIND otherNodes AS otherNode
                        CALL apoc.refactor.mergeNodes([firstNode, otherNode], {
                            properties: 'discard',
                            mergeRels: true
                        })
                        YIELD node
                        RETURN elementId(node)
                    """
                    
                    result, _, _ = self.driver.execute_query(
                        merge_query,
                        {
                            "ids": node_ids,
                            "mergedProps": merged_props
                        },
                        database_=self.neo4j_database,
                    )
                    merged_count += len(result)
            
            total_merged_nodes += merged_count

        return ResolutionStats(
            number_of_nodes_to_resolve=total_entities,
            number_of_created_nodes=total_merged_nodes,
        )
    
    def compute_similarity(self, name_a: str, name_b: str, node_type: str) -> float:
        """
        Use LLM to compute similarity between two entity names of the same type
        Returns a similarity score between 0.0 and 1.0
        """
        if not self.llm:
            # Fallback to simple string matching if no LLM available
            return 1.0 if name_a.lower() == name_b.lower() else 0.0
        
        if node_type == "Component":
            nodeType_based_prompt = f"""These nodes are Components of a Software System, these are services like a Authentication Service, or API Service, things like those.
            Examples:
            Example 1:
            Node1: Authentication Service
            Node2: Auth Service
            Similarity : 1.0

            Example 2:
            Node1: Frontend App
            Node2: Frontend
            Similarity : 0.9

            Example 3:
            Node1: Waitlist Service
            Node2: Waitlist
            Similarity : 0.9

            Example 4:
            Node1: Contact Service
            Node2: Auth Service
            Similarity : 0

            Example 5:
            Node1: API Service
            Node2: Backend Service
            Similarity : 0.5

            Example 6:
            Node1: Frontend Service
            Node2: Database
            Similarity : 0.0

            
            """
        elif node_type == "Technology":
            nodeType_based_prompt = f"""These nodes are Technologies of a Software System, these are programming languages, frameworks, tools, etc. Like Redis, Kafka, MySQL, PostGres DB, Firebase, Next.js, Typescript, Python etc.
            """
        elif node_type == "Decision":
            nodeType_based_prompt = f"""These nodes are Decisions of a Software System, these are decisions like to use a specific technology, or to use a specific service, or to use a specific architecture, etc. These are major level decisions that are made on the project level.
            """
        else:
            nodeType_based_prompt = f"Both nodes are of type: {node_type}"

        # LLM prompt for similarity scoring using names and node type context
        prompt = f"""I have a neo4j graph with some software services and technologies as nodes, i want to merge any similar or duplicate nodes, so for that i need you to tell me if the following nodes are the same or not. 
        For similar nodes give higher , for different nodes give low score.
        {nodeType_based_prompt}
        Node1: {name_a}
        Node2: {name_b}
        give me a score of how similar these nodes are. give me just a number between 0 to 1, 1 is same, 0 is not same, it will act as the scoring of how similar these nodes are. just give this number as output no explanation or anything, just the number"""
        
        try:
            # Call the small LLM to get similarity score
            # response = self.llm.invoke(prompt)
            # Extract the numeric score from response
            # score_text = response.content.strip()
            
            # Make API call to local vLLM server
            import requests
            import json    

            response = self.llm.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            # api_url = "http://localhost:8000/v1/chat/completions"
            # payload = json.dumps({
            # "model": "Qwen/Qwen3-1.7B",
            # "messages": [
            #     {
            #     "role": "system",
            #     "content": "You are a helpful assistant."
            #     },
            #     {
            #     "role": "user",
            #     "content": prompt
            #     }
            # ],
            # "max_tokens": 2000
            # })
            # headers = {
            # 'Content-Type': 'application/json'
            # }

            # response = requests.request("POST", api_url, headers=headers, data=payload)
            
            # response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
            # print(response)
            score_text = response.choices[0].message.content.strip()
            
            print(f"Comparing {node_type} Components:")
            print(f"  Node1: {name_a}")
            print(f"  Node2: {name_b}")
            print(f"Score text: {score_text}")
            print(f"Response: {response}")
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
                    return 1.0 if name_a.lower() == name_b.lower() else 0.0
            
        except Exception as e:
            print(f"⚠️ LLM similarity computation failed: {e}")
            # Fallback to exact matching
            return 1.0 if name_a.lower() == name_b.lower() else 0.0


class CustomKGPipelineConfig(SimpleKGPipelineConfig):
    """Custom pipeline configuration that uses different resolvers based on entity_resolution_type"""

    entity_resolution_type: str = "llm"
    resolution_similarity_threshold: float = 0.8
    slm: AzureOpenAI = None
    
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
