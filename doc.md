# System Architecture: Go-based Worker Service with RabbitMQ and Supabase

## Overview

We are introducing a dedicated Go-based worker service as part of the system architecture to handle background tasks asynchronously, separate from the main application flow.

## Key Components

### Message Queue: RabbitMQ
- **Exchange**: "user_events"
- **Queues**: 
  - "user_verification_queue"
  - "profile_enrichment_queue"
  - "email_validation_queue"
- **Message Structure**: Includes event_type, user_id, email, timestamp, and metadata

### Database: Supabase
- **Users Table**: Contains verification_status, verification_completed_at, profile_enriched_at fields
- **Background_jobs Table**: Tracks job processing with the following fields:
  - id, user_id, job_type, status, attempts, error_message, timestamps

## Workflow

1. **Event Publishing**: Frontend publishes user signup events to RabbitMQ
2. **Job Consumption**: Go worker consumes verification jobs from designated queues
3. **Background Processing**: Worker performs verification tasks including:
   - Email validation
   - Profile enrichment
   - Fraud detection
4. **Database Update**: Worker updates Supabase database with verification results
5. **Status Display**: Frontend displays latest verification status without blocking the initial signup experience

## Architectural Benefits

- **Enhanced User Experience**: Keeps user experience fast and responsive
- **Reliability**: Robust and scalable verification process
- **Asynchronous Processing**: Non-blocking operations with retry mechanisms
- **Scalability**: Horizontal scaling capability with multiple worker instances

## Implementation Features

- **Worker Pools**: Configurable concurrency settings
- **Circuit Breakers**: For external API calls to prevent cascading failures
- **Retry Mechanisms**: Exponential backoff strategy for failed operations
- **Error Handling**: Dead letter queues for managing failed jobs
- **Observability**: Comprehensive logging and monitoring capabilities
- **Data Integrity**: Idempotent job processing to ensure consistency