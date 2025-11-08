import hashlib
import json
from typing import Dict, Any, Set, Optional


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
    
    return output


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
            if getattr(msg, "role", None) == "function":
                content_to_check += f"FUNC_RESULT:{msg.name}:{str(msg.content)[:200]}"
                
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
            if hasattr(msg, 'content') and msg.content:
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
            if getattr(msg, "role", None) == "function":
                agent_output += f"📤 **Result from `{msg.name}`**: {msg.content}\n\n"

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
                    output += (
                        f"📤 **Result from `{tool_name}`**:\n"
                        f"```json\n{formatted}\n```\n\n"
                    )
                else:
                    output += f"📤 **Result from `{tool_name}`**: {tool_content}\n\n"
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
