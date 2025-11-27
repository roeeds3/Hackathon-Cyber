from fastapi import FastAPI
from api_router import router

app = FastAPI(
    title="Cyber Attack Classifier API",
    description="API for classifying EV charging station cyber attacks",
    version="1.0.0"
)

# Include the router
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Cyber Attack Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "classify": "/classify - POST endpoint for classifying a single charging station data",
            "classify-batch": "/classify-batch - POST endpoint for classifying multiple charging station data in batch",
            "visualize": "/visualize - GET endpoint for getting attack cluster visualization (returns PNG image)"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

