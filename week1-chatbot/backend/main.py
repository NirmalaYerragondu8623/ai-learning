from fastapi import FastAPI

app=FastAPI()

@app.get("/health")
async def root():
    return {"status":"OK","model":"gpt-4o-mini"}