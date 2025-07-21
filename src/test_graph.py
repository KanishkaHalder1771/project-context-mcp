"""
Test script for Neo4j GraphBuilder with sample conversation strings
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from storage.graph import GraphBuilder
from config import settings


def test_simple_conversation():
    """Test with a simple design conversation"""
    
    convo1 = """
    1. Overview
We will build a landing website that allows users to join a waitlist and contact the developers. The system will include:

A Frontend service for UI rendering.

An Auth service for user authentication.

A Communication service for waitlist and contact form management.

All services will reside in a single Next.js application but structured as separate modules using routes and components.

2. Tech Stack
Frontend Framework: Next.js (with App Router)

Styling: Tailwind CSS

Authentication: Firebase Authentication

Database: Supabase (PostgreSQL)

Email Service: SendGrid

Deployment: Vercel

State Management: React Context (for session and global UI state)

3. System Architecture
The system will follow a modular approach:

Frontend Service:

Pages: Landing page, navigation to Auth and Contact pages.

Handles UI for waitlist form and links to Auth/Contact pages.

Submits data via Next.js API routes to backend services.

Auth Service:

API Routes: /api/auth/login, /api/auth/signup

Uses Firebase Authentication for user login/signup.

Generates and verifies Firebase JWT tokens.

Login and Signup pages integrated in frontend.

Communication Service:

API Routes:

/api/communication/waitlist → Save waitlist entries to Supabase.

/api/communication/contact → Save contact messages to Supabase and send email via SendGrid.

Contact page for user message submissions.

Server-side validation for all forms.

4. Data Flow
Waitlist:

User submits name + email → API route → Supabase table waitlist.

Contact:

User submits message → API route → Save to Supabase table contacts → Trigger SendGrid email to developers.

Auth:

User signs up or logs in → Firebase issues JWT → Stored in HttpOnly cookie for session management.

Protected routes verify token via Firebase Admin SDK in API routes.

5. Folder Structure
bash
Copy
Edit
/app
  /landing        -> Landing page components
  /auth           -> Login & Signup pages
  /contact        -> Contact form page
/api
  /auth
    login.ts
    signup.ts
  /communication
    waitlist.ts
    contact.ts
/lib
  firebase.ts     -> Firebase config
  supabase.ts     -> Supabase client
  sendgrid.ts     -> Email service helper
/components       -> UI components
6. Deployment
Vercel will handle deployment and CI/CD.

Environment variables for Firebase, Supabase, and SendGrid will be managed in Vercel’s dashboard.

7. Security
JWT stored in HttpOnly cookies to prevent XSS.

API routes will validate input and check authentication when required.

Supabase service keys are server-side only.
    """
    convo2 = "We will also have a Go worker. So from the frontend someone signs up for the waitlist we will run a Background verification job using this Go worker, this when done will update the supabase database with the users status and details." 
    convo3 = "So the Communication service will produce a message to RabbitMq which will be consumed by the go worker"
    
    conversation = convo3
    print("Testing simple conversation:")
    print(conversation)
    print("-" * 50)
    
    return conversation


def main():
    """Run simple conversation test"""
    
    print("🧪 Neo4j GraphBuilder Simple Test")
    print("=" * 50)
    
    if not settings.azure_openai_endpoint:
        print("⚠️  Please set AZURE_OPENAI_ENDPOINT in .env file")
        return
    
    if not settings.azure_openai_api_key:
        print("⚠️  Please set AZURE_OPENAI_API_KEY in .env file")
        return
        
    if not settings.azure_openai_deployment_name:
        print("⚠️  Please set AZURE_OPENAI_DEPLOYMENT_NAME in .env file")
        return
    
    # Check Azure OpenAI configuration
    print(f"🔗 Using Azure OpenAI Endpoint: {settings.azure_openai_endpoint}")
    print(f"🤖 Using Deployment: {settings.azure_openai_deployment_name}")
    print(f"📋 Using API Version: {settings.azure_openai_api_version}")
    
    try:
        # Test with LLM-based resolver (uses small 3B model)
        with GraphBuilder(entity_resolution_type="llm", resolution_similarity_threshold=0.7) as graph_builder:
            
            print("🚀 Building graph for simple conversation")
            
            # Check database connectivity first
            try:
                test_query = "RETURN 1 as test"
                graph_builder.query_graph(test_query)
                print("✅ Neo4j connection successful")
            except Exception as e:
                print(f"❌ Neo4j connection failed: {e}")
                return
            
            # Get the conversation text
            conversation_text = test_simple_conversation()
            
            # Build the graph
            try:
                print("🔨 Starting graph building...")
                graph_builder.build_graph_from_text_sync(conversation_text)
                print("✅ Graph building completed")
            except Exception as e:
                print(f"❌ Graph building failed: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # Check what was created
            new_nodes = graph_builder.query_graph("MATCH (n) RETURN count(n) as node_count")
            print(f"📊 Total nodes created: {new_nodes[0]['node_count'] if new_nodes else 0}")
            
            # Query results
            components = graph_builder.get_components()
            print(f"Components ({len(components)}): {[c['name'] for c in components]}")
            
            technologies = graph_builder.get_technologies()
            print(f"Technologies ({len(technologies)}): {[t['name'] for t in technologies]}")
            
            print("✅ Test completed!\n")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 