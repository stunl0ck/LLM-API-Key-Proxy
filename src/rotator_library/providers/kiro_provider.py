# -*- coding: utf-8 -*-

"""
Kiro Provider - AWS CodeWhisperer/Kiro API integration for LLM-API-Key-Proxy.

Provides Claude models (Opus 4.5, Sonnet 4.5, etc.) via Kiro's internal API.
Supports OAuth refresh token authentication and OpenAI-to-Kiro format conversion.

Based on: https://github.com/jwadow/kiro-openai-gateway (AGPL-3.0)
Adapted for: https://github.com/Mirrowel/LLM-API-Key-Proxy

Key features:
- Refresh token authentication (auto-refresh when expiring)
- OpenAI ↔ Kiro format conversion for messages, tools, and tool results
- AWS SSE stream parsing
- Smart payload management (tool limiting, history trimming) to avoid Kiro API limits
"""

import json
import os
import asyncio
import uuid
import re
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

import httpx
import litellm
import logging

from .provider_interface import ProviderInterface, UsageResetConfigDef

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Kiro API endpoints by region
# Use templates with region interpolation (matching jwadow's implementation)
def _get_kiro_refresh_url(region: str) -> str:
    return f"https://prod.{region}.auth.desktop.kiro.dev/refreshToken"

def _get_kiro_api_host(region: str) -> str:
    return f"https://codewhisperer.{region}.amazonaws.com"

def _get_kiro_q_host(region: str) -> str:
    return f"https://q.{region}.amazonaws.com"

# Token refresh threshold (seconds before expiry)
TOKEN_REFRESH_THRESHOLD = 600  # 10 minutes (match jwadow)

# Model ID mappings (OpenAI model name -> Kiro internal model ID)
# Updated to match jwadow's MODEL_MAPPING
KIRO_MODEL_MAPPING = {
    # Claude Opus 4.5 - top model
    "claude-opus-4-5": "claude-opus-4.5",
    "claude-opus-4-5-20251101": "claude-opus-4.5",
    # Claude Haiku 4.5 - fast model
    "claude-haiku-4-5": "claude-haiku-4.5",
    "claude-haiku-4.5": "claude-haiku-4.5",
    # Claude Sonnet 4.5 - improved model
    "claude-sonnet-4-5": "CLAUDE_SONNET_4_5_20250929_V1_0",
    "claude-sonnet-4-5-20250929": "CLAUDE_SONNET_4_5_20250929_V1_0",
    # Claude Sonnet 4 - balanced model
    "claude-sonnet-4": "CLAUDE_SONNET_4_20250514_V1_0",
    "claude-sonnet-4-20250514": "CLAUDE_SONNET_4_20250514_V1_0",
    # Claude 3.7 Sonnet - legacy
    "claude-3-7-sonnet-20250219": "CLAUDE_3_7_SONNET_20250219_V1_0",
}

# Available models for /v1/models endpoint
KIRO_AVAILABLE_MODELS = [
    "kiro/claude-opus-4-5",
    "kiro/claude-sonnet-4-5",
    "kiro/claude-sonnet-4",
    "kiro/claude-haiku-4-5",
    "kiro/claude-3-7-sonnet-20250219",
]

# Payload limits for smart truncation
MAX_TOOLS = int(os.getenv("KIRO_MAX_TOOLS", "15"))  # Max tool definitions to send
MAX_HISTORY_TURNS = int(os.getenv("KIRO_MAX_HISTORY_TURNS", "20"))  # Max conversation turns
TOOL_DESCRIPTION_MAX_LENGTH = int(os.getenv("KIRO_TOOL_DESC_MAX_LENGTH", "500"))

# Streaming timeouts
FIRST_TOKEN_TIMEOUT = float(os.getenv("KIRO_FIRST_TOKEN_TIMEOUT", "15"))
FIRST_TOKEN_MAX_RETRIES = int(os.getenv("KIRO_FIRST_TOKEN_RETRIES", "2"))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_internal_model_id(model: str) -> str:
    """Convert OpenAI model name to Kiro internal model ID."""
    # Strip provider prefix if present
    clean_model = model.split("/")[-1] if "/" in model else model
    return KIRO_MODEL_MAPPING.get(clean_model, clean_model)


def extract_text_content(content: Any) -> str:
    """
    Extract text content from various OpenAI content formats.
    
    Supports:
    - String: "Hello, world!"
    - List: [{"type": "text", "text": "Hello"}]
    - None: empty message
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif "text" in item:
                    text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    return str(content)


def generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return str(uuid.uuid4())


# =============================================================================
# KIRO AUTH MANAGER
# =============================================================================

@dataclass
class KiroCredentials:
    """Kiro authentication credentials."""
    refresh_token: str
    access_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    profile_arn: Optional[str] = None
    region: str = "us-east-1"
    email: Optional[str] = None  # For credential identification


class KiroAuthManager:
    """
    Manages Kiro API authentication via refresh tokens.
    
    Supports:
    - Environment variable configuration
    - Automatic token refresh when expiring
    - Thread-safe token access
    """
    
    def __init__(
        self,
        refresh_token: str,
        profile_arn: Optional[str] = None,
        region: str = "us-east-1",
        email: Optional[str] = None
    ):
        self.credentials = KiroCredentials(
            refresh_token=refresh_token,
            profile_arn=profile_arn,
            region=region,
            email=email
        )
        self._lock = asyncio.Lock()
        self._http_client: Optional[httpx.AsyncClient] = None
    
    @property
    def refresh_url(self) -> str:
        return _get_kiro_refresh_url(self.credentials.region)
    
    @property
    def api_host(self) -> str:
        return _get_kiro_api_host(self.credentials.region)
    
    @property
    def q_host(self) -> str:
        return _get_kiro_q_host(self.credentials.region)
    
    @property
    def profile_arn(self) -> Optional[str]:
        return self.credentials.profile_arn
    
    def is_token_expiring_soon(self) -> bool:
        """Check if token expires within threshold."""
        if not self.credentials.expires_at:
            return True
        threshold = datetime.now(timezone.utc) + timedelta(seconds=TOKEN_REFRESH_THRESHOLD)
        return self.credentials.expires_at <= threshold
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client
    
    async def _refresh_token_request(self) -> None:
        """Perform token refresh request to Kiro API."""
        client = await self._get_http_client()
        
        payload = {
            "refreshToken": self.credentials.refresh_token
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        lib_logger.debug(f"Refreshing Kiro token at {self.refresh_url}")
        
        response = await client.post(
            self.refresh_url,
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        
        data = response.json()
        
        if "accessToken" not in data:
            raise ValueError("Token refresh response missing accessToken")
        
        self.credentials.access_token = data["accessToken"]
        
        # Parse expiry if provided
        if "expiresAt" in data:
            try:
                self.credentials.expires_at = datetime.fromisoformat(
                    data["expiresAt"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                # Default to 1 hour from now
                self.credentials.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        else:
            self.credentials.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        
        # Update profile ARN if returned
        if "profileArn" in data and not self.credentials.profile_arn:
            self.credentials.profile_arn = data["profileArn"]
        
        lib_logger.info(f"Kiro token refreshed, expires at {self.credentials.expires_at}")
    
    async def get_access_token(self) -> str:
        """Get valid access token, refreshing if necessary."""
        async with self._lock:
            if not self.credentials.access_token or self.is_token_expiring_soon():
                await self._refresh_token_request()
            return self.credentials.access_token
    
    async def force_refresh(self) -> str:
        """Force token refresh (e.g., after 403 error)."""
        async with self._lock:
            self.credentials.access_token = None
            await self._refresh_token_request()
            return self.credentials.access_token
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()


# =============================================================================
# OPENAI TO KIRO CONVERTERS
# =============================================================================

class KiroConverter:
    """
    Converts OpenAI API format to Kiro API format.
    
    Handles:
    - Message format conversion
    - Tool definitions and results
    - Smart payload truncation for Kiro limits
    """
    
    @staticmethod
    def convert_tools(tools: Optional[List[Dict]], max_tools: int = MAX_TOOLS) -> List[Dict]:
        """
        Convert OpenAI tools to Kiro toolSpecification format.
        
        Applies smart truncation:
        - Limits total tools to max_tools
        - Truncates long descriptions
        """
        if not tools:
            return []
        
        # Limit number of tools
        limited_tools = tools[:max_tools]
        if len(tools) > max_tools:
            lib_logger.warning(f"Truncating tools from {len(tools)} to {max_tools}")
        
        kiro_tools = []
        for tool in limited_tools:
            if tool.get("type") != "function":
                continue
            
            func = tool.get("function", {})
            description = func.get("description", "") or ""
            
            # Truncate long descriptions
            if len(description) > TOOL_DESCRIPTION_MAX_LENGTH:
                description = description[:TOOL_DESCRIPTION_MAX_LENGTH - 3] + "..."
                lib_logger.debug(f"Truncated description for tool '{func.get('name')}'")
            
            kiro_tools.append({
                "toolSpecification": {
                    "name": func.get("name", ""),
                    "description": description,
                    "inputSchema": {"json": func.get("parameters", {})}
                }
            })
        
        return kiro_tools
    
    @staticmethod
    def convert_messages(
        messages: List[Dict],
        model_id: str,
        max_history: int = MAX_HISTORY_TURNS
    ) -> tuple[List[Dict], Dict]:
        """
        Convert OpenAI messages to Kiro history and current message format.
        
        Returns:
            Tuple of (history, current_message_dict)
        """
        if not messages:
            raise ValueError("No messages to convert")
        
        # Extract system prompt
        system_prompt = ""
        non_system_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt += extract_text_content(msg.get("content")) + "\n"
            else:
                non_system_messages.append(msg)
        system_prompt = system_prompt.strip()
        
        # Merge adjacent messages and convert tool messages
        merged = KiroConverter._merge_adjacent_messages(non_system_messages)
        
        if not merged:
            raise ValueError("No messages after processing")
        
        # Apply history limit (keep most recent turns)
        if len(merged) > max_history + 1:
            dropped = len(merged) - max_history - 1
            merged = merged[dropped:]
            lib_logger.warning(f"Truncated conversation history by {dropped} turns")
        
        # Split into history and current message
        history_messages = merged[:-1] if len(merged) > 1 else []
        current_msg = merged[-1]
        
        # Inject system prompt into first message
        if system_prompt and history_messages:
            first_msg = history_messages[0]
            if "userInputMessage" in first_msg:
                original = first_msg["userInputMessage"].get("content", "")
                first_msg["userInputMessage"]["content"] = f"{system_prompt}\n\n{original}"
        
        # Build current message
        current_content = extract_text_content(current_msg.get("content", ""))
        if system_prompt and not history_messages:
            current_content = f"{system_prompt}\n\n{current_content}"
        
        if not current_content:
            current_content = "Continue"
        
        current_message = {
            "content": current_content,
            "modelId": model_id,
            "origin": "AI_EDITOR",
        }
        
        return history_messages, current_message
    
    @staticmethod
    def _merge_adjacent_messages(messages: List[Dict]) -> List[Dict]:
        """Merge adjacent messages with same role and convert tool messages."""
        if not messages:
            return []
        
        # First pass: convert tool messages to user messages with tool_results
        processed = []
        pending_tool_results = []
        
        for msg in messages:
            role = msg.get("role")
            
            if role == "tool":
                tool_result = {
                    "content": [{"text": extract_text_content(msg.get("content", "")) or "(empty)"}],
                    "status": "success",
                    "toolUseId": msg.get("tool_call_id", "")
                }
                pending_tool_results.append(tool_result)
            else:
                if pending_tool_results:
                    # Create user message with tool results
                    processed.append({
                        "userInputMessage": {
                            "content": "",
                            "modelId": "",  # Will be filled later
                            "origin": "AI_EDITOR",
                            "userInputMessageContext": {"toolResults": pending_tool_results.copy()}
                        }
                    })
                    pending_tool_results.clear()
                
                # Convert to Kiro format
                if role == "user":
                    processed.append({
                        "userInputMessage": {
                            "content": extract_text_content(msg.get("content", "")),
                            "modelId": "",
                            "origin": "AI_EDITOR",
                        }
                    })
                elif role == "assistant":
                    assistant_msg = {
                        "content": extract_text_content(msg.get("content", ""))
                    }
                    
                    # Handle tool calls
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        tool_uses = []
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                func = tc.get("function", {})
                                try:
                                    args = json.loads(func.get("arguments", "{}"))
                                except json.JSONDecodeError:
                                    args = {}
                                tool_uses.append({
                                    "name": func.get("name", ""),
                                    "input": args,
                                    "toolUseId": tc.get("id", "")
                                })
                        if tool_uses:
                            assistant_msg["toolUses"] = tool_uses
                    
                    processed.append({"assistantResponseMessage": assistant_msg})
        
        # Handle remaining tool results
        if pending_tool_results:
            processed.append({
                "userInputMessage": {
                    "content": "",
                    "modelId": "",
                    "origin": "AI_EDITOR",
                    "userInputMessageContext": {"toolResults": pending_tool_results}
                }
            })
        
        return processed
    
    @staticmethod
    def build_payload(
        messages: List[Dict],
        model: str,
        tools: Optional[List[Dict]],
        profile_arn: str,
        conversation_id: Optional[str] = None
    ) -> Dict:
        """
        Build complete Kiro API payload from OpenAI request.
        
        Applies all smart truncation and conversion logic.
        """
        model_id = get_internal_model_id(model)
        conv_id = conversation_id or generate_conversation_id()
        
        # Convert messages
        history, current_message = KiroConverter.convert_messages(messages, model_id)
        
        # Convert tools (with limits)
        kiro_tools = KiroConverter.convert_tools(tools)
        if kiro_tools:
            if "userInputMessageContext" not in current_message:
                current_message["userInputMessageContext"] = {}
            current_message["userInputMessageContext"]["tools"] = kiro_tools
        
        # Build payload
        payload = {
            "conversationState": {
                "chatTriggerType": "MANUAL",
                "conversationId": conv_id,
                "currentMessage": {
                    "userInputMessage": current_message
                }
            }
        }
        
        if history:
            payload["conversationState"]["history"] = history
        
        if profile_arn:
            payload["profileArn"] = profile_arn
        
        return payload


# =============================================================================
# KIRO RESPONSE PARSER (AWS Event Stream Binary Format)
# =============================================================================

def find_matching_brace(text: str, start_pos: int) -> int:
    """Find matching closing brace accounting for nesting and strings."""
    if start_pos >= len(text) or text[start_pos] != '{':
        return -1
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(start_pos, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return i
    
    return -1


class AwsEventStreamParser:
    """
    Parser for AWS Event Stream binary format.
    
    Kiro returns events in binary SSE format with patterns like:
    {"content": "text"}
    {"name": "tool", "toolUseId": "..."}
    """
    
    EVENT_PATTERNS = [
        ('{"content":', 'content'),
        ('{"name":', 'tool_start'),
        ('{"input":', 'tool_input'),
        ('{"stop":', 'tool_stop'),
        ('{"usage":', 'usage'),
        ('{"contextUsagePercentage":', 'context_usage'),
    ]
    
    def __init__(self):
        self.buffer = ""
        self.last_content: Optional[str] = None
        self.current_tool_call: Optional[Dict] = None
        self.tool_calls: List[Dict] = []
    
    def feed(self, chunk: bytes) -> List[Dict]:
        """Add chunk to buffer and return parsed events."""
        try:
            self.buffer += chunk.decode('utf-8', errors='ignore')
        except Exception:
            return []
        
        events = []
        
        while True:
            earliest_pos = -1
            earliest_type = None
            
            for pattern, event_type in self.EVENT_PATTERNS:
                pos = self.buffer.find(pattern)
                if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
                    earliest_pos = pos
                    earliest_type = event_type
            
            if earliest_pos == -1:
                break
            
            json_end = find_matching_brace(self.buffer, earliest_pos)
            if json_end == -1:
                break  # Incomplete JSON, wait for more data
            
            json_str = self.buffer[earliest_pos:json_end + 1]
            self.buffer = self.buffer[json_end + 1:]
            
            try:
                data = json.loads(json_str)
                event = self._process_event(data, earliest_type)
                if event:
                    events.append(event)
            except json.JSONDecodeError:
                lib_logger.debug(f"Failed to parse Kiro JSON: {json_str[:100]}")
        
        return events
    
    def _process_event(self, data: dict, event_type: str) -> Optional[Dict]:
        """Process a parsed event."""
        if event_type == 'content':
            content = data.get('content', '')
            if content and content != self.last_content:
                self.last_content = content
                return {"type": "content", "data": content}
        
        elif event_type == 'tool_start':
            # Finalize previous tool call if any
            if self.current_tool_call:
                self._finalize_tool_call()
            
            input_data = data.get('input', '')
            input_str = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data or '')
            
            self.current_tool_call = {
                "id": data.get('toolUseId', f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": data.get('name', ''),
                    "arguments": input_str
                }
            }
            
            if data.get('stop'):
                self._finalize_tool_call()
        
        elif event_type == 'tool_input':
            if self.current_tool_call:
                input_data = data.get('input', '')
                input_str = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data or '')
                self.current_tool_call['function']['arguments'] += input_str
        
        elif event_type == 'tool_stop':
            if self.current_tool_call and data.get('stop'):
                self._finalize_tool_call()
        
        elif event_type == 'usage':
            return {"type": "usage", "data": data.get('usage', 0)}
        
        elif event_type == 'context_usage':
            return {"type": "context_usage", "data": data.get('contextUsagePercentage', 0)}
        
        return None
    
    def _finalize_tool_call(self):
        """Finalize current tool call and add to list."""
        if self.current_tool_call:
            # Normalize arguments to JSON string
            args = self.current_tool_call['function']['arguments']
            if isinstance(args, str) and args.strip():
                try:
                    parsed = json.loads(args)
                    self.current_tool_call['function']['arguments'] = json.dumps(parsed)
                except json.JSONDecodeError:
                    self.current_tool_call['function']['arguments'] = "{}"
            else:
                self.current_tool_call['function']['arguments'] = "{}"
            
            self.tool_calls.append(self.current_tool_call)
            self.current_tool_call = None
    
    def get_tool_calls(self) -> List[Dict]:
        """Get all collected tool calls, finalizing any pending ones."""
        if self.current_tool_call:
            self._finalize_tool_call()
        return self.tool_calls


class KiroStreamParser:
    """
    Parses Kiro's AWS SSE stream format and converts to OpenAI format.
    Uses AwsEventStreamParser for binary stream parsing.
    """
    
    @staticmethod
    async def parse_stream(
        response: httpx.Response,
        model: str
    ) -> AsyncGenerator[Dict, None]:
        """
        Parse Kiro SSE stream and yield OpenAI-formatted chunks.
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(datetime.now(timezone.utc).timestamp())
        parser = AwsEventStreamParser()
        
        async for chunk in response.aiter_bytes():
            events = parser.feed(chunk)
            
            for event in events:
                if event["type"] == "content":
                    yield {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": event["data"]},
                            "finish_reason": None
                        }]
                    }
        
        # Get any tool calls
        tool_calls = parser.get_tool_calls()
        
        # Final chunk with finish_reason
        final_delta = {}
        if tool_calls:
            final_delta["tool_calls"] = tool_calls
        
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": final_delta,
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }]
        }


# =============================================================================
# MAIN PROVIDER CLASS
# =============================================================================

class KiroProvider(ProviderInterface):
    """
    Kiro provider for Claude models via AWS CodeWhisperer/Kiro API.
    
    Supports:
    - Claude Opus 4.5, Sonnet 4.5, Sonnet 4, Haiku 4.5
    - OAuth refresh token authentication
    - Full tool calling support (with smart payload limits)
    - Streaming and non-streaming modes
    """
    
    skip_cost_calculation = True  # Free tier, no cost tracking needed
    default_rotation_mode = "sequential"  # Use one credential until exhausted
    provider_env_name = "kiro"
    
    # Tier configuration
    tier_priorities = {
        "default": 1,  # Single tier for now
    }
    default_tier_priority = 1
    
    # Usage tracking (Kiro may have daily/weekly limits)
    usage_reset_configs = {
        "default": UsageResetConfigDef(
            window_seconds=86400,  # 24 hours
            mode="credential",
            description="Daily reset",
            field_name="daily"
        )
    }
    
    # Credential storage: path -> KiroAuthManager
    _auth_managers: Dict[str, KiroAuthManager] = {}
    _init_lock = asyncio.Lock()
    
    def has_custom_logic(self) -> bool:
        return True
    
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Return available Kiro models."""
        return KIRO_AVAILABLE_MODELS.copy()
    
    async def initialize_credentials(self, credential_paths: List[str]) -> None:
        """Initialize auth managers for all credentials."""
        async with self._init_lock:
            for path in credential_paths:
                if path not in self._auth_managers:
                    try:
                        auth = self._create_auth_manager(path)
                        if auth:
                            self._auth_managers[path] = auth
                            lib_logger.info(f"Initialized Kiro credential: {path}")
                    except Exception as e:
                        lib_logger.error(f"Failed to initialize Kiro credential {path}: {e}")
    
    def _create_auth_manager(self, credential_path: str) -> Optional[KiroAuthManager]:
        """Create auth manager from credential path.
        
        Supports:
        - Virtual paths: env://kiro/0, env://kiro/1, etc.
        - Direct env vars: KIRO_REFRESH_TOKEN, KIRO_1_REFRESH_TOKEN
        - File paths: /path/to/credentials.json
        """
        # Handle virtual path format from CredentialManager: env://kiro/N
        if credential_path.startswith("env://kiro/"):
            index = credential_path.split("/")[-1]
            
            # Check for numbered format: KIRO_N_REFRESH_TOKEN
            if index == "0":
                # Legacy single credential format
                refresh_token = os.getenv("KIRO_REFRESH_TOKEN")
                profile_arn = os.getenv("KIRO_PROFILE_ARN")
                region = os.getenv("KIRO_REGION", "us-east-1")
                email = os.getenv("KIRO_EMAIL")
            else:
                # Numbered credential format
                refresh_token = os.getenv(f"KIRO_{index}_REFRESH_TOKEN")
                profile_arn = os.getenv(f"KIRO_{index}_PROFILE_ARN")
                region = os.getenv(f"KIRO_{index}_REGION", "us-east-1")
                email = os.getenv(f"KIRO_{index}_EMAIL")
            
            if not refresh_token:
                lib_logger.warning(f"No refresh token found for credential: {credential_path}")
                return None
            
            return KiroAuthManager(
                refresh_token=refresh_token,
                profile_arn=profile_arn,
                region=region,
                email=email
            )
        
        # Handle direct env var format: KIRO_REFRESH_TOKEN_N
        if credential_path.startswith("KIRO_"):
            refresh_token = os.getenv(credential_path)
            if not refresh_token:
                return None
            
            # Get corresponding profile ARN and region
            suffix = credential_path.replace("KIRO_REFRESH_TOKEN", "")
            profile_arn = os.getenv(f"KIRO_PROFILE_ARN{suffix}")
            region = os.getenv(f"KIRO_REGION{suffix}", "us-east-1")
            email = os.getenv(f"KIRO_EMAIL{suffix}")
            
            return KiroAuthManager(
                refresh_token=refresh_token,
                profile_arn=profile_arn,
                region=region,
                email=email
            )
        
        # File-based credential (JSON file)
        try:
            with open(credential_path, 'r') as f:
                data = json.load(f)
            
            return KiroAuthManager(
                refresh_token=data.get("refreshToken", ""),
                profile_arn=data.get("profileArn"),
                region=data.get("region", "us-east-1"),
                email=data.get("email")
            )
        except Exception as e:
            lib_logger.error(f"Failed to load Kiro credentials from {credential_path}: {e}")
            return None
    
    def _get_auth_manager(self, credential: str) -> Optional[KiroAuthManager]:
        """Get auth manager for a credential."""
        return self._auth_managers.get(credential)
    
    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle chat completion request via Kiro API.
        """
        credential_path = kwargs.pop("credential_identifier", "")
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools")
        stream = kwargs.get("stream", False)
        
        # Get auth manager
        auth = self._get_auth_manager(credential_path)
        if not auth:
            raise ValueError(f"No auth manager for credential: {credential_path}")
        
        # Get access token
        access_token = await auth.get_access_token()
        profile_arn = auth.profile_arn or ""
        
        # Build Kiro payload with smart truncation
        payload = KiroConverter.build_payload(
            messages=messages,
            model=model,
            tools=tools,
            profile_arn=profile_arn
        )
        
        # Make request
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.amazon.eventstream" if stream else "application/json",
        }
        
        api_url = f"{auth.q_host}/api/conversations/chat"
        
        lib_logger.debug(f"Kiro API request to {api_url}, model={model}, stream={stream}")
        
        if stream:
            return self._stream_completion(client, api_url, headers, payload, model, auth)
        else:
            return await self._non_stream_completion(client, api_url, headers, payload, model, auth)
    
    async def _stream_completion(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict,
        payload: Dict,
        model: str,
        auth: KiroAuthManager
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Handle streaming completion."""
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code == 403:
                # Try refreshing token
                new_token = await auth.force_refresh()
                headers["Authorization"] = f"Bearer {new_token}"
                async with client.stream("POST", url, json=payload, headers=headers) as retry_response:
                    retry_response.raise_for_status()
                    async for chunk in KiroStreamParser.parse_stream(retry_response, model):
                        yield self._to_litellm_chunk(chunk)
            elif response.status_code != 200:
                error_text = await response.aread()
                raise httpx.HTTPStatusError(
                    f"Kiro API error: {error_text.decode()}",
                    request=response.request,
                    response=response
                )
            else:
                async for chunk in KiroStreamParser.parse_stream(response, model):
                    yield self._to_litellm_chunk(chunk)
    
    async def _non_stream_completion(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict,
        payload: Dict,
        model: str,
        auth: KiroAuthManager
    ) -> litellm.ModelResponse:
        """Handle non-streaming completion."""
        # For non-streaming, we still use streaming internally and collect
        headers["Accept"] = "application/vnd.amazon.eventstream"
        
        chunks = []
        async for chunk in self._stream_completion(client, url, headers, payload, model, auth):
            chunks.append(chunk)
        
        # Combine chunks into single response
        return self._combine_chunks(chunks, model)
    
    def _to_litellm_chunk(self, chunk: Dict) -> litellm.ModelResponse:
        """Convert chunk dict to litellm ModelResponse."""
        # Pass the dict directly - ModelResponse accepts dict-based structure
        return litellm.ModelResponse(**chunk)
    
    def _combine_chunks(self, chunks: List, model: str) -> litellm.ModelResponse:
        """Combine streaming chunks into a single response."""
        content_parts = []
        tool_calls = []
        finish_reason = "stop"
        
        for chunk in chunks:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                # Access delta safely - it might be a dict or an object
                delta = getattr(choice, 'delta', None)
                if delta:
                    delta_content = getattr(delta, 'content', None) or (delta.get('content') if isinstance(delta, dict) else None)
                    if delta_content:
                        content_parts.append(delta_content)
                    delta_tool_calls = getattr(delta, 'tool_calls', None) or (delta.get('tool_calls') if isinstance(delta, dict) else None)
                    if delta_tool_calls:
                        tool_calls.extend(delta_tool_calls)
                fr = getattr(choice, 'finish_reason', None)
                if fr:
                    finish_reason = fr
        
        content = "".join(content_parts)
        
        # Build response dict
        response_dict = {
            "id": chunks[0].id if chunks and hasattr(chunks[0], 'id') else f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": chunks[0].created if chunks and hasattr(chunks[0], 'created') else int(datetime.now(timezone.utc).timestamp()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls if tool_calls else None
                },
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        return litellm.ModelResponse(**response_dict)
    
    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """Get tier name for credential."""
        return "default"

