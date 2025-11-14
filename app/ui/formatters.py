import hashlib
import json
import re
from typing import Dict, Any, Set, Optional, List, Tuple


TOOL_CALL_START_MARKER = "<!--TOOL_CALL_START-->"
TOOL_CALL_END_MARKER = "<!--TOOL_CALL_END-->"
TOOL_BLOCK_META_PREFIX = "<!--TOOL_BLOCK_META:"
TOOL_BLOCK_META_SUFFIX = "-->"
_TOOL_BLOCK_PATTERN = re.compile(
    rf"{TOOL_CALL_START_MARKER}(.*?){TOOL_CALL_END_MARKER}",
    re.DOTALL
)


def pretty_print_tool_call(name: str, args: Dict[str, Any]) -> str:
    """Format tool call information for display."""
    output = f"🔧 Tool: `{name}`\n\n"
    
    for key, value in args.items():
        if key == "code" or (isinstance(value, str) and value.strip().startswith(("import", "def", "#"))):
            output += f"**📦 Args:** `{key}`:\n"
            output += f"```python\n{value}\n```\n"
        elif isinstance(value, (dict, list)):
            output += f"**📦 Args:** `{key}`:\n"
            output += f"```json\n{json.dumps(value, indent=2)}\n```\n"
        else:
            output += f"**📦 Args:** `{key}`: `{value}`\n\n"
    
    return _wrap_tool_block(output, kind="call", source=name)


def _wrap_tool_block(content: str, *, kind: str = "call", source: Optional[str] = None) -> str:
    """Wrap any tool-related block so the UI can relocate it."""
    body = content.strip()
    if not body:
        return ""
    meta_parts = [f"kind={kind}"]
    if source:
        meta_parts.append(f"source={source}")
    metadata = f"{TOOL_BLOCK_META_PREFIX}{'|'.join(meta_parts)}{TOOL_BLOCK_META_SUFFIX}\n"
    return f"{TOOL_CALL_START_MARKER}\n{metadata}{body}\n{TOOL_CALL_END_MARKER}\n"


def _parse_tool_block_metadata(block: str) -> tuple:
    """Extract metadata and cleaned content from a tool block."""
    kind = "call"
    source = None
    content = block.strip()
    if content.startswith(TOOL_BLOCK_META_PREFIX):
        end_idx = content.find(TOOL_BLOCK_META_SUFFIX)
        if end_idx != -1:
            meta_segment = content[len(TOOL_BLOCK_META_PREFIX):end_idx]
            for part in meta_segment.split("|"):
                if "=" in part:
                    key, value = part.split("=", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "kind" and value:
                        kind = value
                    elif key == "source" and value:
                        source = value
            content = content[end_idx + len(TOOL_BLOCK_META_SUFFIX):].lstrip()
    return kind, source, content


def split_content_with_tool_blocks(markdown: Optional[str]) -> List[Dict[str, Any]]:
    """Split markdown into text/tool segments preserving chronological order."""
    if not markdown:
        return []

    segments: List[Dict[str, Any]] = []
    last_end = 0

    for match in _TOOL_BLOCK_PATTERN.finditer(markdown):
        start, end = match.span()
        if start > last_end:
            text_segment = markdown[last_end:start].strip()
            if text_segment:
                segments.append({"type": "text", "content": text_segment})
        tool_block = match.group(1).strip()
        if tool_block:
            kind, source, cleaned = _parse_tool_block_metadata(tool_block)
            segments.append({
                "type": "tool",
                "kind": kind,
                "source": source,
                "content": cleaned
            })
        last_end = end

    trailing = markdown[last_end:].strip()
    if trailing:
        segments.append({"type": "text", "content": trailing})

    return segments


def extract_tool_call_blocks(markdown: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """Remove tool call blocks from markdown and return remaining text and extracted blocks."""
    if not markdown:
        return markdown, []

    segments = split_content_with_tool_blocks(markdown)
    text_parts = [segment["content"] for segment in segments if segment.get("type") == "text"]
    tool_blocks = [segment["content"] for segment in segments if segment.get("type") == "tool"]
    cleaned = "\n\n".join(text_parts)
    return cleaned, tool_blocks


def _derive_message_id(msg) -> Optional[str]:
    """Get a stable identifier for messages; fall back for tool outputs."""
    msg_id = getattr(msg, "id", None)
    if msg_id:
        return msg_id

    if getattr(msg, "type", None) == "tool":
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            return f"tool_call:{tool_call_id}"

        # Last resort: synthesize a key from tool name + content snapshot
        name = getattr(msg, "name", "tool")
        content = getattr(msg, "content", "")
        signature = f"{name}:{repr(content)[:200]}"
        digest = hashlib.sha1(signature.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"tool_signature:{digest}"

    return None


def separate_agent_outputs(chunk, processed_message_ids, processed_content_hashes: Optional[Set[int]] = None):
    """Separate agent outputs into progress (supervisor/sub-agents) vs final (planning/report).
    
    Args:
        chunk: The message chunk to process
        processed_message_ids: Set of already processed message IDs
        processed_content_hashes: Set of content hashes to prevent duplicate content display
        
    Returns:
        tuple: (progress_output, final_output) - strings for progress expander and main display
    """
    if processed_content_hashes is None:
        processed_content_hashes = set()
        
    progress_output = ""
    final_output = ""

    # Handle different chunk formats
    if isinstance(chunk, tuple):
        return "", ""
    elif not isinstance(chunk, dict):
        return "", ""

    # Standard dictionary format
    for agent_name, data in chunk.items():
        if not isinstance(data, dict):
            continue
        
        # Skip internal system nodes from UI display
        if agent_name.lower() in ["human_chat", "__start__", "__end__"]:
            from app.config import logger
            logger.info(f"Skipping UI display for internal node: {agent_name}")
            continue
            
        messages = data.get("messages", [])
        agent_output = ""
        
        for msg in messages:
            msg_id = _derive_message_id(msg)
            
            if msg_id is None or msg_id in processed_message_ids:
                continue
            
            # Skip user/human messages
            if getattr(msg, "type", None) == "human" or getattr(msg, "role", None) == "user":
                processed_message_ids.add(msg_id)
                continue
            
            msg_role = getattr(msg, "role", None)
            msg_type = getattr(msg, "type", None)
            is_tool_result = (msg_role == "function") or (msg_type == "tool")

            # Content-based deduplication - check if we've seen similar content before
            content_to_check = ""
            
            # Include main message content
            if hasattr(msg, 'content') and msg.content:
                if isinstance(msg.content, list):
                    for c in msg.content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            content_to_check += c.get('text', '')
                else:
                    content_to_check = str(msg.content)
            
            # Include tool calls in content hash to prevent tool duplication
            tool_calls = getattr(msg, "tool_calls", None)
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if isinstance(call, dict):
                        name = call.get("name", "unknown_tool")
                        args = call.get("args", {})
                        # Add tool signature to content hash
                        content_to_check += f"TOOL:{name}:{str(args)[:200]}"
            
            # Include function/tool response content
            if is_tool_result:
                result_label = "FUNC_RESULT" if msg_role == "function" else "TOOL_RESULT"
                content_to_check += f"{result_label}:{getattr(msg, 'name', '')}:{str(getattr(msg, 'content', ''))[:200]}"
                
            # Create content hash for deduplication 
            if content_to_check.strip():
                # Use first 300 chars for primary hash
                primary_hash = hash(content_to_check.strip()[:300])
                
                # For messages with tool calls, also check tool-only hash to catch repeated tool usage
                tool_only_hash = None
                if tool_calls and isinstance(tool_calls, list):
                    tool_signature = ""
                    for call in tool_calls:
                        if isinstance(call, dict):
                            name = call.get("name", "unknown_tool")
                            args = call.get("args", {})
                            tool_signature += f"TOOL:{name}:{str(args)[:200]}"
                    if tool_signature:
                        tool_only_hash = hash(tool_signature)
                
                # Check for duplicates
                if primary_hash in processed_content_hashes:
                    # Mark as processed and skip to prevent duplication
                    processed_message_ids.add(msg_id)
                    continue
                
                # Add hashes to tracking
                processed_content_hashes.add(primary_hash)
                if tool_only_hash:
                    processed_content_hashes.add(tool_only_hash)

            # Print AI and Tool messages content
            if hasattr(msg, 'content') and msg.content and not is_tool_result:
                if isinstance(msg.content, list):
                    for c in msg.content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            agent_output += f"{c.get('text')}\n\n"
                else:
                    agent_output += f"{msg.content}\n\n"

            # Tool calls
            tool_calls = getattr(msg, "tool_calls", None)
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if isinstance(call, dict):
                        name = call.get("name", "unknown_tool")
                        args = call.get("args", {})
                        agent_output += pretty_print_tool_call(name, args)

            # Tool response message (optional)
            if is_tool_result:
                raw_content = getattr(msg, "content", "")
                if isinstance(raw_content, (dict, list)):
                    formatted = json.dumps(raw_content, indent=2)
                    result_block = f"```json\n{formatted}\n```\n\n"
                else:
                    result_block = f"{raw_content}\n\n"
                agent_output += _wrap_tool_block(
                    result_block,
                    kind="result",
                    source=getattr(msg, "name", None)
                )

            processed_message_ids.add(msg_id)
        
        # Only add agent header if there's actual content
        if agent_output:
            # Enhanced supervisor content detection - match reconstruction logic
            supervisor_patterns = ["📋 BREAKDOWN:", "⏳ CURRENT:", "✓ COMPLETED:", "📋 REMAINING:", "📋 OVERALL NOTE"]
            is_supervisor_content = any(pattern in agent_output for pattern in supervisor_patterns)
            
            # If supervisor patterns detected, force agent_name to supervisor (like reconstruction does)
            if is_supervisor_content and agent_name.lower() not in ["planning_agent", "report_agent"]:
                agent_name = "supervisor"
                from app.config import logger
                logger.info(f"Supervisor content pattern detected, forcing agent_name to 'supervisor'")
            
            formatted_agent_output = f"\n\n**{agent_name.upper()}**\n"
            formatted_agent_output += "-" * 40 + "\n\n"
            formatted_agent_output += agent_output
            
            # Determine if this goes to progress expander or final output
            # Planning agent and Report agent go to final output (outside expanders)
            # Supervisor and other sub-agents go to progress expander
            if agent_name.lower() in ["planning_agent", "report_agent"]:
                final_output += formatted_agent_output
            else:
                # supervisor, research_agent, data_agent, prediction_agent go to expander
                progress_output += formatted_agent_output

    return progress_output, final_output


def reconstruct_assistant_response(ai_messages):
    """Reconstruct formatted assistant response maintaining chronological order."""
    try:
        output = ""
        last_agent = None
        
        for msg in ai_messages:
            if not hasattr(msg, 'content'):
                continue
                
            msg_type = getattr(msg, "type", None)
            agent_name = getattr(msg, 'name', 'supervisor')
            
            if msg_type == "tool" and last_agent:
                agent_name = last_agent
            elif msg_type == "tool" and not last_agent:
                agent_name = getattr(msg, "name", "tool_results")
            
            # Only add agent header when agent changes (preserves chronological order)
            if agent_name != last_agent:
                output += f"\n\n**{agent_name.upper()}**\n"
                output += "-" * 40 + "\n\n"
                last_agent = agent_name
            
            if msg_type == "tool":
                tool_name = getattr(msg, "name", "tool")
                tool_content = getattr(msg, "content", "")
                
                if isinstance(tool_content, (dict, list)):
                    formatted = json.dumps(tool_content, indent=2)
                    result_block = f"```json\n{formatted}\n```\n\n"
                else:
                    result_block = f"{tool_content}\n\n"
                output += _wrap_tool_block(result_block, kind="result", source=tool_name)
                continue

            # Add message content for AI outputs
            if msg.content:
                if isinstance(msg.content, list):
                    for c in msg.content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            output += f"{c.get('text')}\n\n"
                else:
                    output += f"{msg.content}\n\n"
            
            # Add tool calls initiated by AI message
            tool_calls = getattr(msg, "tool_calls", None)
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if isinstance(call, dict):
                        name = call.get("name", "unknown_tool")
                        args = call.get("args", {})
                        output += pretty_print_tool_call(name, args)
        
        return output.strip()
        
    except Exception as e:
        from app.config import logger
        logger.error(f"Error reconstructing assistant response: {e}")
        return ""
