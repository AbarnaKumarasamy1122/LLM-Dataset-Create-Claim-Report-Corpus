"""
Week 5 - Task 6: Deployment (Optional)
FastAPI application for claim report generation API.

Usage:
    # Start the server
    python scripts/app.py
    
    # Or with uvicorn
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    
    # API Endpoints:
    #   POST /generate_claim - Generate a single claim report
    #   POST /generate_batch - Generate multiple claim reports
    #   GET /health - Health check
    #   GET /model_info - Model information
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
import uvicorn

# Import the claim generator
import sys
sys.path.append(str(Path(__file__).parent))
from claim_report_generation import ClaimReportGenerator

# Initialize FastAPI app
app = FastAPI(
    title="Claim Report Generation API",
    description="AI-powered claim report generation for logistics damage assessment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global generator instance
generator = None

# Pydantic models
class ShipmentMetadata(BaseModel):
    """Shipment metadata for claim generation."""
    shipment_id: str = Field(..., description="Unique shipment identifier")
    damage_type: str = Field(..., description="Type of damage (dent, crushed, wet, scratch, torn_tape, none)")
    severity: str = Field(..., description="Damage severity (low, medium, high, none)")
    image_id: Optional[str] = Field(None, description="Image identifier")
    vendor: Optional[str] = Field(None, description="Vendor name")
    shipment_stage: Optional[str] = Field(None, description="Shipment stage (warehouse, transit, handling, delivered)")
    
    class Config:
        schema_extra = {
            "example": {
                "shipment_id": "SHP-12345",
                "damage_type": "dent",
                "severity": "high",
                "image_id": "IMG-001.png",
                "vendor": "FastShip Logistics",
                "shipment_stage": "transit"
            }
        }

class GenerationConfig(BaseModel):
    """Configuration for text generation."""
    max_length: int = Field(256, description="Maximum length of generated text", ge=50, le=512)
    num_beams: int = Field(4, description="Number of beams for beam search", ge=1, le=10)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.1, le=2.0)

class ClaimRequest(BaseModel):
    """Request for single claim generation."""
    metadata: ShipmentMetadata
    config: Optional[GenerationConfig] = None

class BatchClaimRequest(BaseModel):
    """Request for batch claim generation."""
    metadata_list: List[ShipmentMetadata]
    config: Optional[GenerationConfig] = None

class ClaimResponse(BaseModel):
    """Response with generated claim report."""
    shipment_id: str
    claim_report: str
    generated_at: str
    model_info: Dict[str, str]

class BatchClaimResponse(BaseModel):
    """Response with multiple generated claim reports."""
    results: List[ClaimResponse]
    total: int
    generated_at: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_path: str
    device: str
    loaded: bool

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global generator
    try:
        print("🚀 Starting Claim Report Generation API...")
        generator = ClaimReportGenerator()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️  API will start but /generate_claim will fail")
        generator = None

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns status of the API and model loading state.
    """
    return {
        "status": "healthy" if generator is not None else "degraded",
        "model_loaded": generator is not None,
        "timestamp": datetime.now().isoformat()
    }

# Model info endpoint
@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the loaded model.
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "google/flan-t5-base (fine-tuned)",
        "model_path": str(generator.model_path),
        "device": generator.device,
        "loaded": True
    }

# Generate single claim endpoint
@app.post("/generate_claim", response_model=ClaimResponse)
async def generate_claim(request: ClaimRequest):
    """
    Generate a claim report from shipment metadata.
    
    Args:
        request: ClaimRequest with metadata and optional generation config
        
    Returns:
        ClaimResponse with generated report
        
    Example:
        ```
        POST /generate_claim
        {
            "metadata": {
                "shipment_id": "SHP-12345",
                "damage_type": "dent",
                "severity": "high",
                "vendor": "FastShip Logistics",
                "shipment_stage": "transit"
            }
        }
        ```
    """
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Prepare generation kwargs
        gen_kwargs = {}
        if request.config:
            gen_kwargs = {
                'max_length': request.config.max_length,
                'num_beams': request.config.num_beams,
                'temperature': request.config.temperature,
            }
        
        # Generate claim
        metadata_dict = request.metadata.dict(exclude_none=True)
        claim_text = generator.generate_from_dict(metadata_dict, **gen_kwargs)
        
        # Return response
        return {
            "shipment_id": request.metadata.shipment_id,
            "claim_report": claim_text,
            "generated_at": datetime.now().isoformat(),
            "model_info": {
                "model": "google/flan-t5-base (fine-tuned)",
                "device": generator.device
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating claim: {str(e)}")

# Generate batch claims endpoint
@app.post("/generate_batch", response_model=BatchClaimResponse)
async def generate_batch(request: BatchClaimRequest):
    """
    Generate multiple claim reports in batch.
    
    Args:
        request: BatchClaimRequest with list of metadata
        
    Returns:
        BatchClaimResponse with list of generated reports
        
    Example:
        ```
        POST /generate_batch
        {
            "metadata_list": [
                {
                    "shipment_id": "SHP-001",
                    "damage_type": "dent",
                    "severity": "high"
                },
                {
                    "shipment_id": "SHP-002",
                    "damage_type": "wet",
                    "severity": "medium"
                }
            ]
        }
        ```
    """
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Prepare generation kwargs
        gen_kwargs = {}
        if request.config:
            gen_kwargs = {
                'max_length': request.config.max_length,
                'num_beams': request.config.num_beams,
                'temperature': request.config.temperature,
            }
        
        # Generate claims for all metadata
        results = []
        for metadata in request.metadata_list:
            metadata_dict = metadata.dict(exclude_none=True)
            claim_text = generator.generate_from_dict(metadata_dict, **gen_kwargs)
            
            results.append({
                "shipment_id": metadata.shipment_id,
                "claim_report": claim_text,
                "generated_at": datetime.now().isoformat(),
                "model_info": {
                    "model": "google/flan-t5-base (fine-tuned)",
                    "device": generator.device
                }
            })
        
        # Return batch response
        return {
            "results": results,
            "total": len(results),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating batch claims: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Claim Report Generation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model_info",
            "generate_claim": "/generate_claim",
            "generate_batch": "/generate_batch",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "status": "running",
        "model_loaded": generator is not None
    }

# Main entry point
if __name__ == "__main__":
    print("\n" + "="*70)
    print("CLAIM REPORT GENERATION API")
    print("="*70)
    print("\n🚀 Starting server...")
    print("   URL: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("   ReDoc: http://localhost:8000/redoc")
    print("\n💡 Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
