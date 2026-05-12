# setup_indexes.py
from pymongo import MongoClient, ASCENDING, DESCENDING

MONGO_URI = "mongodb://localhost:27017/" 
DATABASE_NAME = "sensor_data_db"
COLLECTION_NAME = "readings"

def setup_indexes():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    print("Setting up indexes...")
    
    # 1. Index on the Sensor ID (for fast lookup by device)
    collection.create_index([("sensor_id", ASCENDING)], name="sensor_id_index")
    print("Created index on sensor_id.")
    
    # 2. Index on the Received Time (for fast time-based sorting/filtering)
    collection.create_index([("received_at", DESCENDING)], name="time_desc_index")
    print("Created index on received_at (Descending).")
    
    client.close()

if __name__ == "__main__":
    setup_indexes()