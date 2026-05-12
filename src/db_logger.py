# db_logger.py

import datetime
import time
from pymongo import MongoClient
# Assuming CONFIG has been defined/imported here or you use constants

# --- CONFIGURATION (Ensure these match your receiver setup) ---
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "sensor_data_db" # Or use a more appropriate name like 'rehab_data'
COLLECTION_NAME = "exercise_sessions"

# --- BULK INSERT GLOBALS ---
BATCH_SIZE = 100            
BATCH_TIME_LIMIT = 5.0      

data_buffer = []           
last_insert_time = time.time() 

# --- DATABASE CONNECTION ---
client = None
collection = None

def initialize_db_connection():
    """Initializes the MongoDB connection once."""
    global client, collection
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping') 
        collection = client[DATABASE_NAME][COLLECTION_NAME]
        print("\n[DB Logger] SUCCESS: Connected to MongoDB for session logging.")
    except Exception as e:
        print(f"\n[DB Logger] WARNING: Could not connect to MongoDB. Logging will be disabled. Error: {e}")
        collection = None # Set to None so handle_log knows to skip insertion

def log_session_data(session_data: dict):
    """
    Manages the buffer for bulk inserting final session data.
    `session_data` should contain all finalized metrics for a task part.
    """
    global data_buffer, last_insert_time, collection
    
    if collection is None:
        return # Skip logging if the connection failed
    
    # Add server-side timestamp and append to buffer
    session_data['logged_at'] = datetime.datetime.now()
    data_buffer.append(session_data)

    current_time = time.time()
    
    # Check insertion conditions (Size OR Time - though usually size for session data)
    if len(data_buffer) >= BATCH_SIZE or (current_time - last_insert_time >= BATCH_TIME_LIMIT):
        
        try:
            result = collection.insert_many(data_buffer)
            print(f"[DB Logger] BULK INSERTED {len(data_buffer)} session records.")
            
            # Reset buffer and timer
            data_buffer = []
            last_insert_time = current_time
            
        except Exception as e:
            print(f"[DB Logger] BULK INSERTION FAILED. Error: {e}")

# Call initialization once when the module is imported
initialize_db_connection()