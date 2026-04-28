import sqlite3
import os

def create_database():
    db_path = "company.db"
    
    # Delete the old one if it exists so we start fresh
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print(f"🏗️ Building new corporate database at {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the Sales table
    cursor.execute('''
    CREATE TABLE sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_name TEXT,
        department TEXT,
        revenue REAL,
        month TEXT
    )
    ''')

    # Inject some sample corporate data
    sample_data = [
        ('Alice', 'Engineering', 50000.00, 'January'),
        ('Bob', 'Sales', 120000.50, 'January'),
        ('Charlie', 'Sales', 95000.00, 'February'),
        ('Diana', 'Marketing', 45000.00, 'February'),
        ('Eve', 'Engineering', 60000.00, 'March'),
        ('Frank', 'Sales', 150000.00, 'March')
    ]
    
    cursor.executemany("""
        INSERT INTO sales (employee_name, department, revenue, month) 
        VALUES (?, ?, ?, ?)
    """, sample_data)
    
    conn.commit()
    conn.close()
    
    print("✅ Database created successfully with 6 records.")

if __name__ == "__main__":
    create_database()