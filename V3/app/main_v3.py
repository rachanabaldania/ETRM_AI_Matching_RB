# app/main_v3.py 
# python -m uvicorn app.main_v3:app --reload
# http://127.0.0.1:8000/docs
from fastapi import FastAPI, HTTPException
from app.managers.matching_manager import MatchingManager
from app.models.schemas import MatchRequest
from app.services.data_service import DataService
import logging
from pathlib import Path
import json
from typing import Dict, Any
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
manager = MatchingManager()

class JSONDocumentRequest(BaseModel):
    """Model for the comprehensive shipping document input"""
    status: str
    message: str
    blob_url: str
    mongo_id: str
    extracted_data: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Enhanced startup with detailed logging"""
    global etrm_df
    
    logger.info("Starting ETRM data loading process...")
    
    # Verify basic file access first
    test_path = r"C:\Users\RachanaBaldania\OneDrive - RandomTrees\Rachana_Code\ETRM_AI_Matching_RB\V3\data\ETRM_Data.xlsx"
    try:
        abs_path = Path(test_path).absolute()
        logger.info(f"Checking file at: {abs_path}")
        
        if not abs_path.exists():
            logger.error(f"File not found at: {abs_path}")
            raise RuntimeError("ETRM data file not found at specified location")
            
        if abs_path.stat().st_size == 0:
            logger.error("File exists but is empty")
            raise RuntimeError("ETRM data file is empty")
            
    except Exception as e:
        logger.error(f"File access check failed: {e}")
        raise

    # Load the data
    etrm_df = DataService.load_etrm_data()
    
    if etrm_df is None:
        error_msg = "Failed to load ETRM data - check logs for details"
        logger.critical(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info(f"ETRM data loaded successfully ({len(etrm_df)} records)")
    app.state.etrm_df = etrm_df  # Store in app state for better access

@app.get("/")
async def root():
    return {
        "message": "ETRM Matching API is running",
        "data_loaded": hasattr(app.state, 'etrm_df')
    }

@app.post("/match")
async def match_product(request: MatchRequest):
    """Endpoint for simple product name matching"""
    if not hasattr(app.state, 'etrm_df'):
        logger.error("ETRM data not loaded when matching request received")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable - ETRM data not loaded"
        )
    
    try:
        logger.info(f"Matching request for: {request.extracted_name}")
        result = manager.find_best_match(request.extracted_name, app.state.etrm_df)
        
        if result is None:
            return {"status": "no_match_found"}
            
        try:
            json.dumps(result)  # Test serialization
            return result
        except (TypeError, ValueError) as e:
            logger.error(f"Data serialization error: {e}")
            cleaned_result = manager._clean_nan_values(result)
            return cleaned_result
            
    except Exception as e:
        logger.error(f"Matching failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Matching failed - please check the input and try again"
        )

@app.post("/match_json_document")
async def match_json_document(request: JSONDocumentRequest):
    """Endpoint for comprehensive shipping document matching"""
    if not hasattr(app.state, 'etrm_df'):
        logger.error("ETRM data not loaded when matching request received")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable - ETRM data not loaded"
        )
    
    try:
        # Extract product name from the complex structure
        product_name = request.extracted_data["ShippingDocument"]["ProductDetails"]["Product"]
        logger.info(f"Extracted product name from document: {product_name}")
        
        # Perform matching with the extracted product name
        result = manager.find_best_match(product_name, app.state.etrm_df)
        
        if result is None:
            return {
                "status": "no_match_found",
                "extracted_product": product_name
            }
            
        try:
            json.dumps(result)
            return {
                "match_result": result,
                "document_info": {
                    "shipper": request.extracted_data["ShippingDocument"]["Shipper"]["Name"],
                    "consignee": request.extracted_data["ShippingDocument"]["Consignee"]["Name"],
                    "product": product_name
                }
            }
        except (TypeError, ValueError) as e:
            logger.error(f"Data serialization error: {e}")
            cleaned_result = manager._clean_nan_values(result)
            return {
                "match_result": cleaned_result,
                "document_info": {
                    "shipper": request.extracted_data["ShippingDocument"]["Shipper"]["Name"],
                    "consignee": request.extracted_data["ShippingDocument"]["Consignee"]["Name"],
                    "product": product_name
                }
            }
            
    except KeyError as e:
        logger.error(f"Missing expected field in document: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document structure - missing required field: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Document matching failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Document processing failed - please check the input and try again"
        )