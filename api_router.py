import logging
import pandas as pd
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from typing import Optional, List
from classify_one import predict_one, compute_severity
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
    severity: Optional[float] = Field(None, description="Severity score (0-100) for attacked nodes, None for safe nodes")


class ChargerDataBatchRequest(BaseModel):
    """Request model for batch processing of multiple charging station sensor data"""
    chargers: List[ChargerDataRequest] = Field(..., description="List of charger data objects")


class BatchClassificationResponse(BaseModel):
    """Response model for batch classification results"""
    total_processed: int = Field(..., description="Total number of chargers processed")
    total_attacked: int = Field(..., description="Total number of chargers identified as attacked")
    results: List[ClassificationResponse] = Field(..., description="List of individual classification results")


@router.post("/classify", response_model=ClassificationResponse)
async def classify_charger(data: ChargerDataRequest):
    """
    Classify a charging station record for cyber attacks.
    
    Accepts sensor data from a charging station and returns a summarized
    classification result indicating if the station is under attack.
    """
    try:
        # Convert Pydantic model to tuple for predict_one
        # Tuple order: (charger_id, current, delta_current, voltage, delta_voltage, 
        #              power_w, expected_load, status_str, location, temperature)
        record_tuple = (
            data.charger_id,
            data.current,
            data.delta_current,
            data.voltage,
            data.delta_voltage,
            data.power_w,
            data.expected_load,
            data.status_str,
            (data.loc_x, data.loc_y),
            data.temperature
        )
        
        # Call predict_one to get classification
        prediction_result = predict_one(record_tuple)
        
        # Extract location coordinates from tuple
        loc_x, loc_y = prediction_result["dropped_fields"]["location"]
        
        # Determine if attacked (predicted_class == 2 means CYBER_ATTACK)
        is_attacked = (prediction_result["predicted_class"] == 2)
        
        # Get provider from input or use default
        provider = data.provider if data.provider is not None else "Unknown"
        
        # Compute severity score for attacked nodes only
        severity = None
        if is_attacked:
            # Build record dict for severity computation
            record_dict = {
                "current": data.current,
                "delta_current": data.delta_current,
                "voltage": data.voltage,
                "delta_voltage": data.delta_voltage,
                "power_w": data.power_w,
                "expected_load": data.expected_load,
                "temperature": data.temperature,
                "status_str": data.status_str
            }
            # Get attack probability from prediction
            p_attack = prediction_result["probabilities"]["CYBER_ATTACK(2)"]
            severity = compute_severity(record_dict, p_attack=p_attack)
        
        # Build response dictionary
        result = {
            "ID": str(data.charger_id),
            "Is_attacked": is_attacked,
            "loc_x": float(loc_x),
            "loc_y": float(loc_y),
            "provider": provider,
            "severity": severity
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


@router.post("/classify-batch", response_model=BatchClassificationResponse)
async def classify_chargers_batch(data: ChargerDataBatchRequest):
    """
    Classify multiple charging station records for cyber attacks in a single request.
    
    Accepts a collection of sensor data from multiple charging stations, processes each
    through the ML model, feeds results to the monitor for cluster detection, and returns
    a summary response with all classification results.
    
    After processing, call the /visualize endpoint to view the complete plot with all
    processed chargers and detected attack clusters.
    """
    try:
        results = []
        total_attacked = 0
        
        # Process each charger data item
        for charger_data in data.chargers:
            try:
                # Convert Pydantic model to tuple for predict_one
                # Tuple order: (charger_id, current, delta_current, voltage, delta_voltage, 
                #              power_w, expected_load, status_str, location, temperature)
                record_tuple = (
                    charger_data.charger_id,
                    charger_data.current,
                    charger_data.delta_current,
                    charger_data.voltage,
                    charger_data.delta_voltage,
                    charger_data.power_w,
                    charger_data.expected_load,
                    charger_data.status_str,
                    (charger_data.loc_x, charger_data.loc_y),
                    charger_data.temperature
                )
                
                # Call predict_one to get classification
                prediction_result = predict_one(record_tuple)
                
                # Extract location coordinates from tuple
                loc_x, loc_y = prediction_result["dropped_fields"]["location"]
                
                # Determine if attacked (predicted_class == 2 means CYBER_ATTACK)
                is_attacked = (prediction_result["predicted_class"] == 2)
                
                # Count attacked chargers
                if is_attacked:
                    total_attacked += 1
                
                # Get provider from input or use default
                provider = charger_data.provider if charger_data.provider is not None else "Unknown"
                
                # Compute severity score for attacked nodes only
                severity = None
                if is_attacked:
                    # Build record dict for severity computation
                    record_dict = {
                        "current": charger_data.current,
                        "delta_current": charger_data.delta_current,
                        "voltage": charger_data.voltage,
                        "delta_voltage": charger_data.delta_voltage,
                        "power_w": charger_data.power_w,
                        "expected_load": charger_data.expected_load,
                        "temperature": charger_data.temperature,
                        "status_str": charger_data.status_str
                    }
                    # Get attack probability from prediction
                    p_attack = prediction_result["probabilities"]["CYBER_ATTACK(2)"]
                    severity = compute_severity(record_dict, p_attack=p_attack)
                
                # Build response dictionary
                result = {
                    "ID": str(charger_data.charger_id),
                    "Is_attacked": is_attacked,
                    "loc_x": float(loc_x),
                    "loc_y": float(loc_y),
                    "provider": provider,
                    "severity": severity
                }
                
                # Pass result to Predictor.py for cluster detection and monitoring
                # This updates the monitor's internal state and performs cluster analysis
                # Disable visualization in API context (use /visualize endpoint instead)
                try:
                    monitor.process_json(result, visualize=False)
                except Exception as predictor_error:
                    # Log Predictor errors but don't fail the request
                    # The classification result is still valid and should be included
                    logger.error(f"Error processing result in Predictor for charger {charger_data.charger_id}: {str(predictor_error)}", exc_info=True)
                
                # Add to results list
                results.append(ClassificationResponse(**result))
                
            except ValueError as e:
                # Log individual item errors but continue processing other items
                logger.error(f"Error processing charger {charger_data.charger_id}: {str(e)}", exc_info=True)
                # Optionally, you could include error information in the response
                # For now, we'll skip failed items and continue
                continue
            except Exception as e:
                # Log unexpected errors but continue processing
                logger.error(f"Unexpected error processing charger {charger_data.charger_id}: {str(e)}", exc_info=True)
                continue
        
        # Return batch response
        return BatchClassificationResponse(
            total_processed=len(results),
            total_attacked=total_attacked,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Error in batch classification: {str(e)}", exc_info=True)
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
            node_store=monitor.store,
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

