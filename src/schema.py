from pydantic import BaseModel, Field
from typing import Optional

class SensorReading(BaseModel):
    """Defines the expected schema and constraints for the sensor data."""
    
    # Required Fields
    sensor_id: str = Field(..., min_length=4, max_length=15, description="Unique ID of the sending device")
    temp: float = Field(..., gt=-50.0, lt=100.0, description="Temperature reading in Celsius")
    timestamp_client: str = Field(..., description="ISO formatted time string from the sender")
    
    # Optional Fields
    battery_level: Optional[int] = Field(None, ge=0, le=100, description="Battery percentage (0-100)")