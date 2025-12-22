from os import environ
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


MODEL_URI = environ["MODEL_URI"]


api = FastAPI()

class PredictionRequest(BaseModel):
    inputs: str

@api.get("/health")
async def health():
    return {"status": "ok"}

@api.post("/predict")
async def predict(request: PredictionRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MODEL_URI}/invocations",
            json={"inputs": request.inputs},
            headers={"Content-Type": "application/json"},
        )
        return response.json()
    # return {"prediction": "result_here", "input": request.inputs}

@api.get("/")
async def index():
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: sans-serif; display: flex; min-height: 100vh; margin: 0; }
                .container { max-width: 500px; padding: 20px; }
                input { width: 100%; padding: 8px; margin-bottom: 10px; font-size: 16px; }
                button { width: 100%; padding: 8px; background: #0066cc; color: white; border: none; cursor: pointer; }
                #result { margin-top: 20px; display: none; }
            </style>
            <script>
                async function handleSubmit(e) {
                    e.preventDefault();
                    const r = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({inputs: document.getElementById('input').value})
                    });
                    const d = await r.json();
                    console.log(d);
                    document.getElementById('resultText').textContent = JSON.stringify(d, null, 2);
                    document.getElementById('result').style.display = 'block';
                }
            </script>
        </head>
        <body>
            <div class="container">
                <h1>Prediction</h1>
                <form onsubmit="handleSubmit(event)">
                    <input type="text" id="input" placeholder="Enter text" required>
                    <button type="submit">Submit</button>
                </form>
                <div id="result"><strong>Result:</strong><pre id="resultText"></pre></div>
            </div>
        </body>
        </html>
        """
    )


def main():
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)