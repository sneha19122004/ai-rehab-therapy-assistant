import socket
import json
import time 
import datetime 
import random 

# --- Configuration ---
UDP_IP = "127.0.0.1" 
UDP_PORT = 5005
SENSOR_ID = "FactoryB3-Snsr"
SEND_INTERVAL_SECONDS = 5 
# --- Configuration ---

# Setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"Starting continuous sender for {SENSOR_ID} to {UDP_IP}:{UDP_PORT}...")

while True:
    # Prepare dynamic data
    current_temp = round(random.uniform(20.0, 30.0), 2)
    current_time = datetime.datetime.now().isoformat()
    
    data = {
        "sensor_id": SENSOR_ID, 
        "temp": current_temp, 
        "timestamp_client": current_time,
        "battery_level": random.randint(80, 100)
    }
    
    MESSAGE = json.dumps(data).encode('utf-8')
    
    # Send the data
    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SENT: Temp {current_temp}°C")
    
    # Wait before sending the next packet
    time.sleep(SEND_INTERVAL_SECONDS)