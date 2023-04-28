from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse, HTMLResponse
import functools
import io
import yaml
import uvicorn
from pydantic import BaseModel
from gradio_tools import <<tool-name>>


tool = <<Insert Tool Here>>

plugin_json = <<Insert JSON Here>>

class PostBody(BaseModel):
    query: str

app = FastAPI()


@app.get("/")
def landing_page():
    return HTMLResponse(
        """
        <p>Just a bot to sync data from diffusers gallery please go to
    <a href="https://huggingface.co/spaces/huggingface-projects/diffusers-gallery" target="_blank" rel="noopener noreferrer">https://huggingface.co/spaces/huggingface-projects/diffusers-gallery</a>
    </p>""")


@app.get('/openapi.yaml', include_in_schema=False)
@functools.lru_cache()
def read_openapi_yaml() -> Response:
    openapi_json= app.openapi()
    yaml_s = io.StringIO()
    yaml.dump(openapi_json, yaml_s)
    return Response(yaml_s.getvalue(), media_type='text/yaml')


@app.get("/.well-known/ai-plugin.json")
def ai_plugin(name: str) -> JSONResponse:
    return JSONResponse(plugin_json)


@app.post("/predict")
def predict(body: PostBody) -> JSONResponse:
    return JSONResponse({"output": tool.run(body.query)})


uvicorn.run(app, host="0.0.0.0", port=7860)
