import json
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from app.config import THREAD_IDS_FILE, SQLITE_DB_PATH, logger


def load_thread_ids() -> List[Dict]:
    """Load thread IDs from JSON file."""
    try:
        if THREAD_IDS_FILE.exists():
            with open(THREAD_IDS_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading thread IDs: {e}")
        return []


def save_thread_ids(thread_ids: List[Dict]) -> None:
    """Save thread IDs to JSON file."""
    try:
        with open(THREAD_IDS_FILE, 'w') as f:
            json.dump(thread_ids, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving thread IDs: {e}")


def add_thread_id(thread_id: str, title: Optional[str] = None) -> None:
    """Add a new thread ID to the persistent storage."""
    thread_ids = load_thread_ids()
    
    # Check if thread already exists
    if any(t["thread_id"] == thread_id for t in thread_ids):
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    thread_data = {
        "thread_id": thread_id,
        "created_at": timestamp,
        "title": title or f"Conversation {timestamp}"
    }
    
    thread_ids.append(thread_data)
    save_thread_ids(thread_ids)


def delete_thread_from_database(thread_id: str) -> bool:
    """Delete all database records for a given thread ID from both checkpoints and writes tables."""
    try:
        connection = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
        cursor = connection.cursor()
        
        # Delete from writes table first (due to potential foreign key constraints)
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        writes_deleted = cursor.rowcount
        
        # Delete from checkpoints table
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        checkpoints_deleted = cursor.rowcount
        
        connection.commit()
        
        # Run VACUUM to reclaim disk space and ensure deleted data is completely removed
        cursor.execute("VACUUM")
        
        connection.close()
        
        logger.info(f"Deleted {checkpoints_deleted} checkpoints and {writes_deleted} writes for thread {thread_id}, database vacuumed")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting thread {thread_id} from database: {e}")
        return False


def remove_thread_id(thread_id: str) -> None:
    """Remove a thread ID from persistent storage and database."""
    # Remove from JSON file
    thread_ids = load_thread_ids()
    thread_ids = [t for t in thread_ids if t["thread_id"] != thread_id]
    save_thread_ids(thread_ids)
    
    # Remove from database
    delete_thread_from_database(thread_id)


def update_thread_title(thread_id: str, new_title: str) -> None:
    """Update the title of a thread."""
    thread_ids = load_thread_ids()
    for thread in thread_ids:
        if thread["thread_id"] == thread_id:
            thread["title"] = new_title
            break
    save_thread_ids(thread_ids)


def generate_new_thread_id() -> str:
    """Generate a new unique thread ID."""
    return str(uuid.uuid4())