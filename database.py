"""
database.py - SQLite Database Management for SmartLead AI
--------------------------------------------------------
Handles creation, storage, querying, and bulk insertion of customer leads.
No external database server needed: uses standard Python sqlite3.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Default path to SQLite database file
DB_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DB_DIR / "leads.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database."""
    target_path = db_path or DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initializes the database schema if tables do not exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            service TEXT NOT NULL,
            budget TEXT NOT NULL,
            urgency TEXT NOT NULL,
            message TEXT NOT NULL,
            predicted_category TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            priority_score INTEGER NOT NULL,
            sentiment_score REAL NOT NULL,
            sentiment_label TEXT NOT NULL,
            key_requirements TEXT,
            recommended_action TEXT,
            status TEXT DEFAULT 'New'
        )
    """)
    conn.commit()
    conn.close()


def insert_lead(lead_dict: Dict[str, Any], db_path: Optional[Path] = None) -> int:
    """
    Inserts a single evaluated lead into the database.
    Returns the newly generated lead ID.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    timestamp = lead_dict.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO leads (
            timestamp, name, email, service, budget, urgency,
            message, predicted_category, confidence_score, priority_score,
            sentiment_score, sentiment_label, key_requirements,
            recommended_action, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        lead_dict["name"],
        lead_dict["email"],
        lead_dict["service"],
        lead_dict["budget"],
        lead_dict["urgency"],
        lead_dict["message"],
        lead_dict.get("predicted_category", "Cold"),
        float(lead_dict.get("confidence_score", 0.0)),
        int(lead_dict.get("priority_score", 0)),
        float(lead_dict.get("sentiment_score", 0.0)),
        lead_dict.get("sentiment_label", "Neutral"),
        lead_dict.get("key_requirements", ""),
        lead_dict.get("recommended_action", "Standard follow-up"),
        lead_dict.get("status", "New")
    ))
    
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id


def get_all_leads(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Retrieves all leads from the database as a Pandas DataFrame."""
    conn = get_connection(db_path)
    query = "SELECT * FROM leads ORDER BY priority_score DESC, id DESC"
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def get_leads_filtered(
    category: Optional[str] = None,
    service: Optional[str] = None,
    search_query: Optional[str] = None,
    db_path: Optional[Path] = None
) -> pd.DataFrame:
    """Queries leads with optional filters for category, service, and text search."""
    conn = get_connection(db_path)
    conditions: List[str] = []
    params: List[Any] = []
    
    if category and category != "All":
        conditions.append("predicted_category = ?")
        params.append(category)
        
    if service and service != "All":
        conditions.append("service = ?")
        params.append(service)
        
    if search_query:
        conditions.append("(name LIKE ? OR email LIKE ? OR message LIKE ?)")
        wildcard = f"%{search_query}%"
        params.extend([wildcard, wildcard, wildcard])
        
    sql = "SELECT * FROM leads"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY priority_score DESC, id DESC"
    
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def update_lead_status(lead_id: int, new_status: str, db_path: Optional[Path] = None) -> None:
    """Updates the status (e.g., 'New', 'Contacted', 'Qualified', 'Lost') of a lead."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE leads SET status = ? WHERE id = ?", (new_status, lead_id))
    conn.commit()
    conn.close()


def delete_lead(lead_id: int, db_path: Optional[Path] = None) -> None:
    """Deletes a lead by ID."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM leads WHERE id = ?", (lead_id,))
    conn.commit()
    conn.close()


def bulk_insert_leads(leads_list: List[Dict[str, Any]], db_path: Optional[Path] = None) -> int:
    """Bulk inserts a list of lead dictionaries into SQLite."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    records = []
    for lead in leads_list:
        ts = lead.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        records.append((
            ts,
            lead["name"],
            lead["email"],
            lead["service"],
            lead["budget"],
            lead["urgency"],
            lead["message"],
            lead.get("predicted_category", "Cold"),
            float(lead.get("confidence_score", 0.0)),
            int(lead.get("priority_score", 0)),
            float(lead.get("sentiment_score", 0.0)),
            lead.get("sentiment_label", "Neutral"),
            lead.get("key_requirements", ""),
            lead.get("recommended_action", "Standard follow-up"),
            lead.get("status", "New")
        ))
        
    cursor.executemany("""
        INSERT INTO leads (
            timestamp, name, email, service, budget, urgency,
            message, predicted_category, confidence_score, priority_score,
            sentiment_score, sentiment_label, key_requirements,
            recommended_action, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


if __name__ == "__main__":
    # Self-test block: Verify database creation & record insertion
    print("Testing database.py initialization...")
    init_db()
    
    sample_lead = {
        "name": "Arjun Sharma",
        "email": "arjun@example.com",
        "service": "AI & Machine Learning Solutions",
        "budget": "$10,000 - $25,000",
        "urgency": "Immediate (within 1-2 weeks)",
        "message": "We need an AI lead scoring engine integrated into our CRM immediately.",
        "predicted_category": "Hot",
        "confidence_score": 0.94,
        "priority_score": 92,
        "sentiment_score": 0.85,
        "sentiment_label": "Positive",
        "key_requirements": "AI lead scoring engine, CRM integration",
        "recommended_action": "Assign Senior Account Executive immediately; schedule discovery call within 2 hours.",
        "status": "New"
    }
    
    lead_id = insert_lead(sample_lead)
    print(f"Sample lead inserted successfully with ID: {lead_id}")
    
    df = get_all_leads()
    print(f"Total leads retrieved from SQLite: {len(df)}")
    print(df[["id", "name", "predicted_category", "priority_score"]].to_string())
