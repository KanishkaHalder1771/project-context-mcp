"""
Context Manager Service

Handles all context storage, retrieval, and management operations
using a singleton GraphBuilder instance.
"""

import os
import json
import requests
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
    
    def _get_doc_path(self) -> str:
        """Get the path to the documentation markdown file"""
        # Store doc.md in the project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(project_root, "doc.md")
    
    def _read_existing_doc(self) -> str:
        """Read existing documentation content"""
        doc_path = self._get_doc_path()
        if os.path.exists(doc_path):
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"⚠️ ContextManager: Error reading doc.md: {e}")
                return ""
        return ""
    
    def _write_doc(self, content: str) -> bool:
        """Write content to documentation file"""
        doc_path = self._get_doc_path()
        try:
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ ContextManager: Updated doc.md at {doc_path}")
            return True
        except Exception as e:
            print(f"❌ ContextManager: Error writing doc.md: {e}")
            return False
    
    async def _unify_context_with_llm(self, existing_context: str, new_context: str) -> str:
        """Use LLM to unify existing and new context"""
        try:
            # Temporary URL and API key for testing
            api_url = "https://api.example.com/v1/chat/completions"  # Replace with actual LLM API
            api_key = "dummy-api-key-12345"  # Replace with actual API key
            
            prompt = f"""You are a technical documentation assistant. Your task is to unify and organize software architecture discussions.

EXISTING CONTEXT:
{existing_context}

NEW CONTEXT TO ADD:
{new_context}

Please create a unified, well-organized markdown document that:
1. Combines both contexts seamlessly
2. Removes any redundant information
3. Organizes content logically with proper headings
4. Maintains technical accuracy
5. Uses clear, professional language

Return only the unified markdown content without any additional commentary.  Just the markdown content"""

            payload = {
                "model": "gpt-3.5-turbo",  # Adjust model as needed
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 4000,
                "temperature": 0.3
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            print("🤖 ContextManager: Calling LLM to unify context...")
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                unified_content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                print("✅ ContextManager: LLM successfully unified context")
                return unified_content
            else:
                print(f"⚠️ ContextManager: LLM API error {response.status_code}: {response.text}")
                # Fallback: simple concatenation
                return self._simple_context_merge(existing_context, new_context)
                
        except Exception as e:
            print(f"❌ ContextManager: LLM unification failed: {e}")
            # Fallback: simple concatenation
            return self._simple_context_merge(existing_context, new_context)
    
    def _simple_context_merge(self, existing_context: str, new_context: str) -> str:
        """Fallback method for simple context merging"""
        if not existing_context.strip():
            return f"# Software Architecture Documentation\n\n{new_context}"
        
        return f"{existing_context}\n\n---\n\n## New Context Added\n\n{new_context}"
    
    async def answer_query(self, query: str) -> str:
        """Answer query using the documentation context"""
        try:
            # Read existing doc content
            doc_content = self._read_existing_doc()
            
            # Temporary URL and API key for testing
            api_url = "https://api.example.com/v1/chat/completions"  # Replace with actual LLM API
            api_key = "dummy-api-key-12345"  # Replace with actual API key
            
            prompt = f"""This is what my current project looks like:
{doc_content}

Query: {query}

Answer the query based on the context provided. Just the answer."""

            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 4000,
                "temperature": 0.3
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                return answer
            else:
                return f"Error: API request failed with status {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def store_context(self, text: str) -> Dict[str, Any]:
        try:
            print(f"🔍 ContextManager: Storing context: {text[:100]}...")
            
            # Step 1: Store in graph database
            await self._graph_builder.build_graph_from_text(text)
            print(f"✅ ContextManager: Graph built successfully from text")
            
            # Step 2: Handle markdown documentation
            print("📝 ContextManager: Processing markdown documentation...")
            
            # Read existing doc content
            existing_context = self._read_existing_doc()
            
            # Unify contexts using LLM
            unified_context = await self._unify_context_with_llm(existing_context, text)
            
            # Write unified content back to doc.md
            doc_written = self._write_doc(unified_context)
            
            return {
                'success': True,
                'message': 'Context stored successfully in graph',
                'graph_stored': True,
                'doc_updated': doc_written
            }
            
        except Exception as e:
            print(f"❌ ContextManager: Failed to store context: {str(e)}")
            return {
                'success': False,
                'message': f'Failed to store context: {str(e)}',
                'graph_stored': False,
                'doc_updated': False
            } 