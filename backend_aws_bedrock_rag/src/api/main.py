from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr

# PUBLIC_INTERFACE
class AgentInput(BaseModel):
    """Represents a single AWS Bedrock Agent selector with its metadata."""
    role: constr(strip_whitespace=True, min_length=1) = Field(..., description="Human-readable role/title for the agent (e.g., Gastroenterologist).")
    agent_id: constr(strip_whitespace=True, min_length=1) = Field(..., description="AWS Bedrock Agent ID.")
    alias_id: constr(strip_whitespace=True, min_length=1) = Field(..., description="AWS Bedrock Agent Alias ID to target.")
    region: constr(strip_whitespace=True, min_length=1) = Field(..., description="AWS region where the agent is deployed (e.g., us-east-1).")

# PUBLIC_INTERFACE
class QueryAgentsRequest(BaseModel):
    """Request payload for querying multiple AWS Bedrock Agents."""
    agents: List[AgentInput] = Field(..., description="A list of agent descriptors to query.")
    query: constr(strip_whitespace=True, min_length=1) = Field(..., description="The user question or prompt to send to the LLM.")

# PUBLIC_INTERFACE
class AgentResponse(BaseModel):
    """Response for a single agent invocation."""
    role: str = Field(..., description="Agent role echoed back for context.")
    agent_id: str = Field(..., description="Agent ID echoed back.")
    alias_id: str = Field(..., description="Alias ID echoed back.")
    region: str = Field(..., description="Region echoed back.")
    success: bool = Field(..., description="Whether the Bedrock invocation succeeded.")
    output: Optional[Dict[str, Any]] = Field(None, description="Raw Bedrock response payload or normalized content if available.")
    error: Optional[str] = Field(None, description="Error message if invocation failed.")

# PUBLIC_INTERFACE
class QueryAgentsResponse(BaseModel):
    """Aggregated response containing results for each agent."""
    query: str = Field(..., description="Original query string.")
    results: List[AgentResponse] = Field(..., description="Per-agent results including success/error information.")


openapi_tags = [
    {
        "name": "Health",
        "description": "Service health and readiness probes."
    },
    {
        "name": "Agents",
        "description": "Endpoints for querying AWS Bedrock Agents."
    }
]

app = FastAPI(
    title="AWS Bedrock Agent Query Backend",
    description="FastAPI backend that accepts an 'agents' list and forwards queries to AWS Bedrock Agents per-region, returning structured results.",
    version="0.1.0",
    openapi_tags=openapi_tags
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"], summary="Health Check", description="Simple health check to verify the service is running.", operation_id="health_check")
def health_check():
    """Health check endpoint.
    Returns a JSON object indicating the service is healthy.
    """
    return {"message": "Healthy"}


def _get_bedrock_runtime_client(region: str):
    """Create a Bedrock Runtime client for a specific AWS region.

    Note:
    - This function requires AWS credentials to be available in the environment (e.g., via IAM role, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY).
    - The project must include 'boto3' in requirements for actual runtime usage.
    """
    try:
        import boto3  # Imported here to avoid import error if not needed in certain environments
    except Exception as e:
        raise RuntimeError("boto3 is required to call AWS Bedrock. Please ensure boto3 is installed.") from e

    # Using bedrock-agent-runtime for InvokeAgent, but some setups use 'bedrock-runtime' for model invocations.
    # We will use 'bedrock-agent-runtime' to align with agent_id/alias_id usage.
    return boto3.client("bedrock-agent-runtime", region_name=region)


def _invoke_bedrock_agent(region: str, agent_id: str, alias_id: str, query: str) -> Dict[str, Any]:
    """Invoke the specified Bedrock Agent and return the raw response as a dict.

    This function uses the Bedrock Agent Runtime's InvokeAgent API.

    Raises:
        Exception: Propagates exceptions for the caller to handle and wrap.
    """
    client = _get_bedrock_runtime_client(region)

    # The Bedrock Agents InvokeAgent API typically requires:
    # - agentId
    # - agentAliasId
    # - sessionId (we can generate a stateless/session value)
    # - inputText (user query)
    # For demo purposes, we'll generate a simple session ID.
    import uuid
    session_id = str(uuid.uuid4())

    # Some SDKs return a streaming response. For simplicity, we aggregate the content.
    response = client.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        inputText=query
    )

    # The response structure can include 'completion' or streamed 'chunk' content.
    # To make this generic, return the raw dict-like metadata and any text payload if available.
    # boto3 returns a dict with a 'completion' (dict) or 'responseStream' (streaming).
    # If streaming is used, you'd iterate response["completion"]["content"] or similar.
    # Here, we attempt to extract known fields safely.
    result: Dict[str, Any] = {}
    for k in ("sessionId", "invocationId", "contentType", "completion"):
        if k in response:
            result[k] = response[k]

    # Some responses may include plain 'completion' with text
    # Ensure we don't raise if fields are missing
    if "completion" in response and isinstance(response["completion"], dict):
        # Normalize text if present
        text = response["completion"].get("text") or response["completion"].get("content") or None
        if text is not None:
            result["text"] = text

    return result


@app.post(
    "/query_agents",
    tags=["Agents"],
    summary="Query AWS Bedrock Agents",
    description=(
        "Accepts a list of agents (role, agent_id, alias_id, region) and a query string. "
        "For each agent, forwards the query to AWS Bedrock in the specified region and returns aggregated results. "
        "Requires AWS credentials to be configured via environment or IAM role."
    ),
    operation_id="query_agents",
    response_model=QueryAgentsResponse,
    responses={
        200: {"description": "Aggregated results for each requested agent."},
        400: {"description": "Invalid input request."},
        500: {"description": "Server error invoking AWS Bedrock."},
    },
)
def query_agents(payload: QueryAgentsRequest) -> QueryAgentsResponse:
    """Query multiple AWS Bedrock Agents.

    Parameters:
    - payload: QueryAgentsRequest
        - agents: List of agent descriptors containing role, agent_id, alias_id, and region.
        - query: The question or prompt for the LLM.

    Returns:
    - QueryAgentsResponse: Contains the original query and a list of results for each agent, including success status and outputs or errors.

    Environment:
    - Requires valid AWS credentials to be available to the runtime (IAM role, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, etc.).
    """
    # Validate that at least one agent is provided
    if not payload.agents:
        raise HTTPException(status_code=400, detail="At least one agent must be provided.")

    results: List[AgentResponse] = []

    for agent in payload.agents:
        try:
            output = _invoke_bedrock_agent(
                region=agent.region,
                agent_id=agent.agent_id,
                alias_id=agent.alias_id,
                query=payload.query,
            )
            results.append(
                AgentResponse(
                    role=agent.role,
                    agent_id=agent.agent_id,
                    alias_id=agent.alias_id,
                    region=agent.region,
                    success=True,
                    output=output,
                    error=None,
                )
            )
        except Exception as e:
            # Capture per-agent failure while allowing others to continue
            results.append(
                AgentResponse(
                    role=agent.role,
                    agent_id=agent.agent_id,
                    alias_id=agent.alias_id,
                    region=agent.region,
                    success=False,
                    output=None,
                    error=str(e),
                )
            )

    return QueryAgentsResponse(query=payload.query, results=results)
