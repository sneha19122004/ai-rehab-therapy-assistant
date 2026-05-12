import socket
import json
import datetime
from time import sleep # Used for robust connection retries
from pymongo import MongoClient
from pydantic import BaseModel, Field, ValidationError # Pydantic imports
from typing import Optional
# --- Global Setup (Add these lines near the top of your receiver script) ---
import time # Ensure you import time
import datetime # Ensure you import datetime

BATCH_SIZE = 100            # Insert after 100 documents are buffered
BATCH_TIME_LIMIT = 5.0      # Insert after 5 seconds, whichever comes first

data_buffer = []            # Global list to hold documents waiting for insertion
last_insert_time = time.time() # Tracks the last time a bulk insert was performed

# --- 1. CONFIGURATION ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "sensor_data_db"
COLLECTION_NAME = "readings"

# --- 2. GLOBAL VARIABLES ---
# These will be initialized in the main execution block
mongo_client = None
collection = None
listen_sock = None


# --- 3. PYDANTIC SCHEMA DEFINITION (Crucial for Validation) ---
# NOTE: If you put this in schema.py, you would use: from schema import SensorReading
class SensorReading(BaseModel):
    """Defines the expected schema for the incoming sensor data."""
    
    # Required Fields
    sensor_id: str = Field(..., min_length=4, max_length=15)
    temp: float = Field(..., gt=-50.0, lt=100.0)
    timestamp_client: str = Field(..., description="ISO formatted time string from the sender")
    
    # Optional Fields
    battery_level: Optional[int] = Field(None, ge=0, le=100)


# --- 4. DATA STORAGE FUNCTION ---
def handle_buffer(payload_dict: dict):
    """
    Manages the buffer, inserting data only when the size limit or time limit is met.
    """
    global data_buffer, last_insert_time, collection 
    
    # 1. Add server-side timestamp and append to buffer
    payload_dict['received_at'] = datetime.datetime.now()
    data_buffer.append(payload_dict)

    current_time = time.time()
    
    # 2. Check insertion conditions (Size OR Time)
    if len(data_buffer) >= BATCH_SIZE or (current_time - last_insert_time >= BATCH_TIME_LIMIT):
        
        try:
            # The high-performance insert command
            result = collection.insert_many(data_buffer)
            
            print(f"|--- BULK INSERTED {len(data_buffer)} documents. First ID: {result.inserted_ids[0]}")
            
            # Reset buffer and timer
            data_buffer = []
            last_insert_time = current_time
            
        except Exception as e:
            # Handle failure: log and try inserting the data again later
            print(f"|--- BULK INSERTION FAILED. {len(data_buffer)} documents lost or need reprocessing. Error: {e}")
# --- 5. INITIALIZATION FUNCTIONS ---
def initialize_mongodb(uri, db_name, collection_name, retries=5):
    """Establishes a persistent connection to MongoDB with retry logic."""
    global mongo_client, collection
    
    for attempt in range(retries):
        try:
            mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Test connection immediately
            mongo_client.admin.command('ping') 
            
            collection = mongo_client[db_name][collection_name]
            
            print("SUCCESS: Connected to MongoDB and established collection reference.")
            return True
        
        except Exception as e:
            print(f"ATTEMPT {attempt + 1}/{retries}: MongoDB connection failed. Retrying in 2 seconds...")
            print(f"Details: {e}")
            sleep(2)
    
    print("FATAL: Could not connect to MongoDB after multiple retries. Exiting.")
    return False

def initialize_udp_socket(ip, port):
    """Creates and binds the UDP listening socket."""
    global listen_sock
    try:
        listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listen_sock.bind((ip, port))
        
        print(f"--- UDP Listener Started ---")
        print(f"Listening on {ip}:{port}...")
        return True
    except Exception as e:
        print(f"FATAL: Could not bind UDP socket to {ip}:{port}. Check if another process is using the port. Error: {e}")
        return False


# --- 6. MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    
    # Initialize services
    if not initialize_mongodb(MONGO_URI, DATABASE_NAME, COLLECTION_NAME):
        exit(1)
        
    if not initialize_udp_socket(UDP_IP, UDP_PORT):
        exit(1)

    try:
        # This is your requested loop, now fully integrated!
        while True:
            # Blocks until a packet is received
            data, addr = listen_sock.recvfrom(1024) 
            
            try:
                decoded_data = data.decode('utf-8')
                
                # Check 1: Must be valid JSON
                json_payload = json.loads(decoded_data)
                
                # Check 2: Must conform to Pydantic schema
                # **The core of the validation logic**
                validated_data = SensorReading(**json_payload)
                
                # Convert Pydantic object back to a standard dictionary
                payload_dict = validated_data.model_dump()
                
                print(f"\nReceived (VALID): {payload_dict['sensor_id']} | Temp: {payload_dict['temp']}")

                # Store the clean data using the single insert function
                handle_buffer(payload_dict)

            except json.JSONDecodeError:
                print(f"\nERROR: Received malformed JSON from {addr}. Packet discarded.")
            
            except ValidationError as e:
                # Catches errors like missing fields, wrong data types, or out-of-range values
                print(f"\nVALIDATION ERROR: Data failed schema check from {addr}.")
                print(f"DETAILS: {e.errors()}")
            
            except Exception as e:
                # Catches any other runtime errors inside the processing block
                print(f"\nGeneral processing error on packet from {addr}: {e}")

    except KeyboardInterrupt:
        print("\nReceiver shutdown initiated by user (Ctrl+C).")
    
    finally:
        if listen_sock:
            listen_sock.close()
            print("UDP socket closed.")
        if mongo_client:
            mongo_client.close()
            print("MongoDB connection closed.")