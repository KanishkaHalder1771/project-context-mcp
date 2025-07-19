"""
Pydantic data models for type safety and validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RelationshipType(Enum):
    """Types of relationships between contexts"""
    DEPENDS_ON = "depends_on"
    AFFECTS = "affects"
    USES = "uses"
    DECIDES = "decides"

# Neo4j Schema Definitions
NEO4J_NODE_TYPES = [
    {"label": "Component", "properties": [{"name": "name", "type": "STRING"}, {"name": "description", "type": "STRING"}]},
    {"label": "Technology", "properties": [{"name": "name", "type": "STRING"}]},
    {"label": "Decision", "properties": [{"name": "name", "type": "STRING"}, {"name": "description", "type": "STRING"}]}
]

NEO4J_RELATIONSHIP_TYPES = [
    {"label": "USES", "properties": [{"name": "strength", "type": "FLOAT"}]},
    {"label": "DECIDES", "properties": [{"name": "strength", "type": "FLOAT"}]},
    {"label": "DEPENDS_ON", "properties": [{"name": "strength", "type": "FLOAT"}]}
]

NEO4J_PATTERNS = [
    ("Component", "USES", "Technology"),
    ("Decision", "DECIDES", "Component"),
    ("Decision", "DECIDES", "Technology"),
    ("Component", "DEPENDS_ON", "Component")
]

NEO4J_SCHEMA = {
    "node_types": NEO4J_NODE_TYPES,
    "relationship_types": NEO4J_RELATIONSHIP_TYPES,
    "patterns": NEO4J_PATTERNS,
    "additional_node_types": False,
    "additional_relationship_types": False,
    "additional_properties": False
}