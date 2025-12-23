import sqlite3
import json
import datetime
from typing import List, Optional, Dict, Any, Tuple
import hashlib

class PersonDatabase:
    def __init__(self, db_path: str = "persons.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Persons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                face_encoding TEXT NOT NULL,
                registration_date TEXT NOT NULL,
                last_seen TEXT,
                total_detections INTEGER DEFAULT 0,
                metadata TEXT,
                UNIQUE(name)
            )
        ''')
        
        # Detection logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                detection_time TEXT NOT NULL,
                confidence REAL,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_person(self, name: str, face_encoding: List[float], metadata: Dict = None) -> int:
        """Add a new person to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO persons (name, face_encoding, registration_date, metadata)
                VALUES (?, ?, ?, ?)
            ''', (
                name,
                json.dumps(face_encoding),
                datetime.datetime.now().isoformat(),
                json.dumps(metadata or {})
            ))
            
            person_id = cursor.lastrowid
            conn.commit()
            return person_id
        except sqlite3.IntegrityError:
            raise ValueError(f"Person with name '{name}' already exists")
        finally:
            conn.close()
    
    def update_person(self, person_id: int, name: str = None, metadata: Dict = None):
        """Update person information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        values = []
        
        if name:
            updates.append("name = ?")
            values.append(name)
        
        if metadata is not None:
            updates.append("metadata = ?")
            values.append(json.dumps(metadata))
        
        if updates:
            values.append(person_id)
            cursor.execute(f'''
                UPDATE persons 
                SET {', '.join(updates)}
                WHERE id = ?
            ''', values)
            
        conn.commit()
        conn.close()
    
    def delete_person(self, person_id: int):
        """Delete a person from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        cursor.execute("DELETE FROM detection_logs WHERE person_id = ?", (person_id,))
        
        conn.commit()
        conn.close()
    
    def get_all_persons(self) -> List[Dict]:
        """Get all persons from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM persons 
            ORDER BY last_seen DESC, name ASC
        ''')
        
        persons = []
        for row in cursor.fetchall():
            person = dict(row)
            person['face_encoding'] = json.loads(person['face_encoding'])
            person['metadata'] = json.loads(person['metadata']) if person['metadata'] else {}
            persons.append(person)
        
        conn.close()
        return persons
    
    def get_person_by_id(self, person_id: int) -> Optional[Dict]:
        """Get person by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
        row = cursor.fetchone()
        
        if row:
            person = dict(row)
            person['face_encoding'] = json.loads(person['face_encoding'])
            person['metadata'] = json.loads(person['metadata']) if person['metadata'] else {}
            conn.close()
            return person
        
        conn.close()
        return None
    
    def log_detection(self, person_id: int, confidence: float = None):
        """Log a detection event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = datetime.datetime.now().isoformat()
        
        # Insert detection log
        cursor.execute('''
            INSERT INTO detection_logs (person_id, detection_time, confidence)
            VALUES (?, ?, ?)
        ''', (person_id, current_time, confidence))
        
        # Update person's last_seen and increment detection count
        cursor.execute('''
            UPDATE persons 
            SET last_seen = ?, total_detections = total_detections + 1
            WHERE id = ?
        ''', (current_time, person_id))
        
        conn.commit()
        conn.close()
    
    def get_detection_stats(self, days: int = 7) -> Dict:
        """Get detection statistics for the last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
        
        # Total detections
        cursor.execute('''
            SELECT COUNT(*) FROM detection_logs 
            WHERE detection_time > ?
        ''', (cutoff_date,))
        total_detections = cursor.fetchone()[0]
        
        # Unique persons detected
        cursor.execute('''
            SELECT COUNT(DISTINCT person_id) FROM detection_logs 
            WHERE detection_time > ?
        ''', (cutoff_date,))
        unique_persons = cursor.fetchone()[0]
        
        # Most frequent visitors
        cursor.execute('''
            SELECT p.name, COUNT(*) as count 
            FROM detection_logs dl
            JOIN persons p ON dl.person_id = p.id
            WHERE dl.detection_time > ?
            GROUP BY p.id
            ORDER BY count DESC
            LIMIT 5
        ''', (cutoff_date,))
        top_visitors = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_detections': total_detections,
            'unique_persons': unique_persons,
            'top_visitors': top_visitors
        }
    
    def search_persons(self, query: str) -> List[Dict]:
        """Search persons by name"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM persons 
            WHERE name LIKE ?
            ORDER BY name ASC
        ''', (f'%{query}%',))
        
        persons = []
        for row in cursor.fetchall():
            person = dict(row)
            person['face_encoding'] = json.loads(person['face_encoding'])
            person['metadata'] = json.loads(person['metadata']) if person['metadata'] else {}
            persons.append(person)
        
        conn.close()
        return persons