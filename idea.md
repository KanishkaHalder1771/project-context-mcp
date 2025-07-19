# Project Context MCP Server

## Core Concept
Create an MCP server that acts as a "project memory" for coding agents like Cursor/Claude - allowing users to simply say **"add this to context"** and automatically storing, categorizing, and connecting architectural context, design decisions, API specs, and workflows to solve the problem of losing project context across sessions.

### Key Features
- **Zero-friction storage**: Just say "add this to context" - no manual categorization needed
- **Smart auto-categorization**: AI-powered category detection and assignment  
- **Intelligent relationships**: Automatic discovery of connections between contexts
- **Semantic search**: Find contexts by meaning, not just keywords
- **Session continuity**: Never lose project context again

## Problem Statement
When working on complex projects (e.g., video hosting service), developers often lose context about:
- General architecture and design decisions
- API specifications and how they work
- Authentication flows and user journeys
- Business rules and domain logic
- Dependencies and their relationships

Opening a new session with Cursor/Claude loses all this valuable context.

## Solution Architecture

### Hybrid Storage Approach

#### 1. Knowledge Graph (Relationships & Structure)
- **Nodes**: APIs, Features, Components, Design Decisions, User Flows, Auth Methods
- **Edges**: "depends_on", "implements", "conflicts_with", "evolved_from", "related_to"
- **Tools**: Neo4j, ArangoDB, or simple graph structure in SQLite

Example relationships:
```
[User Registration API] --implements--> [JWT Auth Flow]
[Video Upload Feature] --depends_on--> [S3 Storage Service]
[Payment System] --conflicts_with--> [Free Tier Logic]
```

#### 2. Vector Database (Semantic Search)
- **Embeddings**: Convert all context text to vectors for semantic similarity
- **Tools**: Chroma (local), Pinecone, Weaviate, or FAISS
- **Chunks**: Break contexts into meaningful segments for better retrieval

### Smart Auto-Categorization System

#### Core Categories (Auto-detected)
- **Architecture**: System design, tech stack decisions, patterns used
- **APIs**: Endpoints, request/response formats, authentication methods  
- **User Flows**: Step-by-step user journeys, wireframes, business logic
- **Database**: Schema designs, relationships, data models
- **Auth**: Authentication/authorization flows, user roles, permissions
- **Business Rules**: Domain logic, validation rules, edge cases
- **Dependencies**: Why certain libraries were chosen, configuration details
- **Infrastructure**: Deployment, scaling, monitoring, DevOps
- **UI/UX**: Frontend components, styling, user interface decisions

#### Auto-Detection Process
1. **Content Analysis**: Extract keywords and technical terms
2. **Vector Similarity**: Compare with existing categorized contexts
3. **Confidence Scoring**: Assign confidence levels to category suggestions
4. **Smart Defaults**: 
   - High confidence (>0.8): Auto-apply category
   - Medium confidence (0.5-0.8): Suggest category to user
   - Low confidence (<0.5): Ask user for clarification
5. **Custom Categories**: Allow project-specific categories when needed

#### User Experience
- **Simple Input**: "Add this to context: JWT implementation with Redis storage"
- **Auto-Processing**: System detects "auth" category with 92% confidence
- **Smart Response**: "✅ Saved as 'JWT Authentication with Redis' in 'auth' category"

## Streamlined MCP Tools (5 Tools Total)

**Design Philosophy**: Simple, intuitive tools that LLMs can easily understand and use. No complex parameters or context IDs - just natural language descriptions.

### 1. **`list_categories`**
- **Purpose**: Show LLM what context categories exist in this project
- **Params**: None
- **Output**: Array of available categories with counts
  ```json
  {
    "categories": [
      {"name": "auth", "count": 5, "description": "Authentication and authorization"},
      {"name": "api", "count": 12, "description": "API endpoints and integrations"}, 
      {"name": "database", "count": 8, "description": "Database schemas and queries"},
      {"name": "ui_ux", "count": 3, "description": "User interface components"}
    ]
  }
  ```

### 2. **`store_context`**
- **Purpose**: Universal storage tool - handles both creating new contexts and updating existing ones automatically
- **Params**: 
  - `content` (string, required): The context content to store
  - `category` (string, optional): Target category ("auto" for auto-detection)
- **Output**: Action taken, final context title, category assigned, any duplicates found
- **Smart Features**:
  - Auto-detects category if not specified
  - Finds similar existing contexts and offers update/merge options
  - Creates relationships with related contexts automatically
  - Generates meaningful titles from content

**Example Usage:**
```
store_context(
  content="JWT authentication with Google OAuth and username/password fallback",
  category="auth"
)
```

### 3. **`get_context`**
- **Purpose**: Retrieve contexts about a specific topic
- **Params**: 
  - `topic` (string, required): What you're looking for (e.g., "authentication setup")
  - `category` (string, optional): Filter by category for better precision
  - `max_results` (int, optional): Maximum contexts to return (default: 5)
- **Output**: Matching contexts with full content, titles, and metadata
- **Search Method**: Combines semantic similarity + keyword matching + category filtering

**Example Usage:**
```
get_context(topic="JWT authentication", category="auth")
→ Returns all authentication-related contexts
```

### 4. **`get_related_contexts`**
- **Purpose**: Find everything connected to or related to a specific topic
- **Params**: 
  - `topic` (string, required): Central topic to explore
  - `category` (string, optional): Starting category filter
  - `depth` (int, optional): How deep to search relationships (default: 2)
- **Output**: Related contexts with relationship types and connection paths
- **Discovery Method**: Uses knowledge graph traversal + semantic similarity

**Example Usage:**
```
get_related_contexts(topic="user authentication", category="auth")
→ Returns: OAuth setup, session management, password hashing, login forms, etc.
```

### 5. **`delete_context`**
- **Purpose**: Remove contexts about a specific topic
- **Params**: 
  - `topic` (string, required): Description of what to delete
  - `category` (string, optional): Limit search to specific category  
  - `confirm` (boolean, optional): Skip confirmation prompt (default: false)
- **Output**: Contexts found for deletion, dependency warnings, deletion results
- **Safety Features**: 
  - Shows what will be deleted before doing it
  - Warns about dependent contexts
  - Option to update context instead of deleting

**Example Usage:**
```
delete_context(topic="old PayPal integration", category="api")
→ Finds PayPal contexts, shows dependencies, confirms before deletion
```

## LLM Workflow Pattern

**Typical LLM interaction flow:**

1. **Discover categories**: `list_categories()` to see what's available
2. **Store new info**: `store_context(content="...", category="auth")`  
3. **Retrieve context**: `get_context(topic="authentication", category="auth")`
4. **Explore connections**: `get_related_contexts(topic="auth setup")`
5. **Clean up**: `delete_context(topic="old auth method")`

## Tool Coordination Behind the Scenes

While LLMs only see 5 simple tools, the system internally:
- **Vector DB**: Handles semantic search and similarity detection
- **Knowledge Graph**: Manages relationships and dependencies  
- **Auto-categorizer**: Assigns categories with confidence scoring
- **Duplicate detector**: Prevents redundant contexts
- **Relationship engine**: Builds connections between contexts automatically

The complexity is hidden - LLMs just describe what they want in natural language!

## Key Design Principle: No Context IDs Required

**Users and LLMs never need to know internal context IDs.** All interactions use natural language descriptions and search queries. The system handles the complexity of finding, matching, and updating contexts automatically.

## Example User Workflows

### Simple Context Storage
```
User: "Add this to context: We're using Stripe for payments with webhook validation"
System: ✅ Saved as 'Stripe Payment Integration' in 'api' category (auto-detected)
        🔗 Connected to existing 'Payment Flow' and 'Webhook Security' contexts
```

### Automatic Update Detection
```
User: "Add this to context: JWT tokens now expire in 1 hour instead of 24 hours"
System: 🔍 Found similar context: 'JWT Authentication Setup' (similarity: 89%)
        💡 This looks like an update. Options:
        1. Update existing context
        2. Create separate context  
        3. Merge both contexts
User: "Update existing"
System: ✅ Updated 'JWT Authentication Setup' with new expiration details
```

### Explicit Context Updates
```
User: "Update our authentication setup - we've added 2FA support"
System: 🔍 Found 'JWT Authentication Setup' 
        ✅ Added 2FA information to existing authentication context
        🔗 Auto-connected to 'User Security' and 'Mobile App Login' contexts
```

### Natural Language Exploration
```
User: "Show me how authentication connects to our payment system"
System: 🗺️  Connection found: Authentication → User Sessions → Payment Authorization
        📋 Path: JWT Auth → Redis Sessions → Payment API → Stripe Integration
        🔗 3 hop connection with 'depends_on' relationships
```

### Smart Deletion
```
User: "Delete the old payment context about PayPal - we switched to Stripe"
System: 🔍 Found 'PayPal Payment Integration' 
        ⚠️  This context has 2 dependencies: 'Checkout Flow' and 'Refund Process'
        💡 Options: Delete only | Delete cascade | Update to mention migration
User: "Update to mention migration"
System: ✅ Updated context to document PayPal → Stripe migration
```

## Benefits

1. **Zero Learning Curve**: No need to learn categories or metadata - just describe your context
2. **Semantic Discovery**: Find related contexts even without exact keywords
3. **Relationship Mapping**: See how changing one component affects others
4. **Context Clustering**: Automatically group related architectural decisions
5. **Impact Analysis**: Understand dependencies before making changes
6. **Session Continuity**: Maintain project context across multiple sessions
7. **Smart Organization**: AI handles the boring categorization work

## Implementation Considerations

### Technology Stack
- **Embedding Model**: Local models (sentence-transformers) or API-based (OpenAI)
- **Graph Database**: Start with NetworkX + SQLite, scale to Neo4j
- **Vector Store**: Chroma for local development
- **MCP Framework**: Python or TypeScript SDK

### Scalability
- **Local vs Cloud**: Start local, option for cloud sync
- **Multi-project Support**: Separate context stores per project
- **Versioning**: Handle context evolution over time
- **Performance**: Efficient indexing for large context stores

## Next Steps
1. Choose technology stack
2. Implement basic storage and retrieval
3. Add semantic search capabilities
4. Build graph relationship features
5. Create intuitive MCP tool interface
6. Test with real project scenarios 