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

## Ultra-Simple MCP Tools (4 Tools Total)

**Design Philosophy**: Minimal interface with maximum intelligence. No categories, no complex parameters - just natural language content and queries.

### 1. **`store_context`**
- **Purpose**: Universal storage tool - handles both creating new contexts and updating existing ones automatically
- **Params**: 
  - `content` (string, required): The context content to store
- **Output**: Action taken, final context title, auto-detected category, any duplicates found/merged
- **Smart Features**:
  - Auto-detects category internally (no user input needed)
  - Finds similar existing contexts and auto-merges or updates
  - Creates relationships with related contexts automatically
  - Generates meaningful titles from content

**Example Usage:**
```
store_context(content="JWT authentication with Google OAuth and username/password fallback")
```

### 2. **`get_context`**
- **Purpose**: Retrieve contexts about a specific topic using semantic search
- **Params**: 
  - `query` (string, required): What you're looking for (e.g., "authentication setup", "JWT tokens")
- **Output**: Matching contexts with full content, titles, and relevance scores
- **Search Method**: Pure semantic similarity search across all stored contexts

**Example Usage:**
```
get_context(query="JWT authentication")
→ Returns all authentication-related contexts ranked by relevance
```

### 3. **`get_related_contexts`**
- **Purpose**: Find everything connected to or related to a specific topic
- **Params**: 
  - `query` (string, required): Central topic to explore relationships around
- **Output**: Related contexts with relationship types and connection paths
- **Discovery Method**: Uses knowledge graph traversal + semantic similarity from query

**Example Usage:**
```
get_related_contexts(query="user authentication")
→ Returns: OAuth setup, session management, password hashing, login forms, etc.
```

### 4. **`delete_all_contexts`**
- **Purpose**: Completely clear all stored contexts (nuclear option)
- **Params**: None
- **Output**: Confirmation of total contexts deleted
- **Safety Features**: 
  - Shows total count before deletion
  - Irreversible operation warning

**Example Usage:**
```
delete_all_contexts()
→ Deletes all contexts after confirmation
```

## LLM Workflow Pattern

**Typical LLM interaction flow:**

1. **Store new info**: `store_context(content="JWT authentication with Google OAuth...")`  
2. **Retrieve context**: `get_context(query="authentication")`
3. **Explore connections**: `get_related_contexts(query="auth setup")`
4. **Nuclear reset**: `delete_all_contexts()` (when starting fresh)

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

### Complete Reset
```
User: "Clear all context - starting fresh with new architecture"
System: ⚠️  About to delete ALL contexts (47 total)
        💀 This action is irreversible!
User: "Confirm"
System: ✅ Deleted all 47 contexts. Fresh slate ready!
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