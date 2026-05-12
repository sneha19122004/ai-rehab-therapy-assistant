import socket

UDP_IP = "127.0.0.1" 
UDP_PORT = 5006
UDP_SOCKET = None
try:
    UDP_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # SOCK_DGRAM specifies UDP
    print(f"UDP Sender initialized on {UDP_IP}:{UDP_PORT}")
except Exception as e:
    print(f"Error initializing UDP socket: {e}")
    UDP_SOCKET = None

def send_rep_count(rep_count, exercise_code):
    """Sends a standardized message to Node-RED."""
    if UDP_SOCKET is None:
        return
        
    # Standard format: "REP_COUNT|CODE|NUMBER"
    message = f"REP_COUNT|{exercise_code}|{rep_count}"
    
    try:
        UDP_SOCKET.sendto(message.encode('utf-8'), (UDP_IP, UDP_PORT))
        # print(f"Sent: {message}") # Uncomment for debugging
    except Exception as e:
        print(f"Error sending UDP message: {e}")
