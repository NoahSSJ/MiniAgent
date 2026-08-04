import httpx
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
OLLAMA_URL = "http://127.0.0.1:11434"

class OllamaRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False

@app.get("/models")
async def list_models():
    async with httpx.AsyncClient(trust_env=False) as client:
        resp = await client.get(f"{OLLAMA_URL}/api/tags")
        resp.raise_for_status()
        return resp.json()
    
@app.post("/gen")
async def generate(req: OllamaRequest):
    async with httpx.AsyncClient(timeout=300.0, trust_env=False) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=req.model_dump()
            )
            resp.raise_for_status()
            return resp.json()
    
if __name__ == "__main__":
    uvicorn.run("route:app", host="127.0.0.1", port=8000, reload=True)