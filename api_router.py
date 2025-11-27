import logging
import pandas as pd
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from typing import Optional
from classify_one import predict_one
from Predictor import StreamingAttackMonitor

# Configure logging
logger = logging.getLogger(__name__)

# Create singleton instance of StreamingAttackMonitor
# This maintains state across all API requests for cluster detection
monitor = StreamingAttackMonitor(distance_threshold=300, min_cluster_size=3)

router = APIRouter()


class ChargerDataRequest(BaseModel):
    """Request model for charging station sensor data"""
    charger_id: int = Field(..., description="Charger identifier")
    current: float = Field(..., description="Current reading")
    delta_current: float = Field(..., description="Change in current")
    voltage: float = Field(..., description="Voltage reading")
    delta_voltage: float = Field(..., description="Change in voltage")
    power_w: float = Field(..., description="Power in watts")
    expected_load: float = Field(..., description="Expected load")
    status_str: str = Field(..., description="Status string (e.g., 'CHARGING')")
    loc_x: float = Field(..., description="X coordinate")
    loc_y: float = Field(..., description="Y coordinate")
    temperature: float = Field(..., description="Temperature reading")
    provider: Optional[str] = Field(None, description="Provider name")


class ClassificationResponse(BaseModel):
    """Response model for classification results"""
    ID: str = Field(..., description="Charger ID as string")
    Is_attacked: bool = Field(..., description="Whether the charger is under attack")
    loc_x: float = Field(..., description="X coordinate")
    loc_y: float = Field(..., description="Y coordinate")
    provider: str = Field(..., description="Provider name")


@router.post("/classify", response_model=ClassificationResponse)
async def classify_charger(data: ChargerDataRequest):
    """
    Classify a charging station record for cyber attacks.
    
    Accepts sensor data from a charging station and returns a summarized
    classification result indicating if the station is under attack.
    """
    try:
        # Convert Pydantic model to dict for predict_one
        # Convert separate x, y fields to tuple for location (as expected by predict_one)
        record_dict = {
            "charger_id": data.charger_id,
            "current": data.current,
            "delta_current": data.delta_current,
            "voltage": data.voltage,
            "delta_voltage": data.delta_voltage,
            "power_w": data.power_w,
            "expected_load": data.expected_load,
            "status_str": data.status_str,
            "location": (data.loc_x, data.loc_y),
            "temperature": data.temperature
        }
        
        # Call predict_one to get classification
        prediction_result = predict_one(record_dict)
        
        # Extract location coordinates from tuple
        loc_x, loc_y = prediction_result["dropped_fields"]["location"]
        
        # Determine if attacked (predicted_class == 2 means CYBER_ATTACK)
        is_attacked = (prediction_result["predicted_class"] == 2)
        
        # Get provider from input or use default
        provider = data.provider if data.provider is not None else "Unknown"
        
        # Build response dictionary
        result = {
            "ID": str(data.charger_id),
            "Is_attacked": is_attacked,
            "loc_x": float(loc_x),
            "loc_y": float(loc_y),
            "provider": provider
        }
        
        # Pass result to Predictor.py for cluster detection and monitoring
        # This updates the monitor's internal state and performs cluster analysis
        # Disable visualization in API context (use /visualize endpoint instead)
        try:
            monitor.process_json(result, visualize=False)
        except Exception as predictor_error:
            # Log Predictor errors but don't fail the request
            # The classification result is still valid and should be returned
            logger.error(f"Error processing result in Predictor: {str(predictor_error)}", exc_info=True)
        
        return ClassificationResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/visualize")
async def get_visualization():
    """
    Get the current visualization of attack clusters and at-risk nodes.
    Returns a PNG image showing all nodes, clusters, and at-risk areas.
    """
    try:
        # Get current state from monitor
        all_nodes = monitor.store.get_all_nodes()
        attacked = monitor.store.get_attacked_nodes()
        
        # If no data, return empty visualization
        if all_nodes.empty:
            raise HTTPException(status_code=404, detail="No data available for visualization")
        
        # Generate visualization with image return
        results = monitor.detector.detect(all_nodes, attacked)
        # Ensure clustered_df exists (empty DataFrame if no attacked nodes)
        clustered_df = results.get("clustered_df", pd.DataFrame())
        at_risk_nodes = results.get("at_risk_nodes", pd.DataFrame())
        
        image_bytes = monitor.detector.visualize(
            all_nodes,
            clustered_df,
            at_risk_nodes,
            return_image=True
        )
        
        if image_bytes:
            return Response(content=image_bytes, media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail="Failed to generate visualization")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

