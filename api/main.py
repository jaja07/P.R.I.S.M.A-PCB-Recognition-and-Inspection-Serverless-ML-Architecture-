from fastapi import FastAPI

try:
    from api.endpoints import router
except ModuleNotFoundError:
    from endpoints import router

app = FastAPI(
    title="P.R.I.S.M.A. Inference API",
    description="Serverless API for fault detection on printed circuit boards",
    version="1.0.0"
)

app.include_router(router, prefix="/prisma")

@app.get("/", tags=["Health"])
async def root():
    return {"message": "API is running", "status": "ok"}
    
def main():
    """Entry point to launch the application locally via Uvicorn."""
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
