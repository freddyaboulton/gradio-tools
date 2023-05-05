from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import functools
import io
import yaml
import uvicorn
from pydantic import BaseModel
from gradio_tools import <<tool-name>>
import pathlib
import urllib.parse
import json
import os

tool = <<Insert Tool Here>>

plugin_json = <<Insert JSON Here>>

class PostBody(BaseModel):
    query: str

app = FastAPI()

is_spaces = os.getenv("SYSTEM") == "spaces"

def get_url(request: Request):
    port = f":{request.url.port}" if request.url.port else ""
    return f"{'https' if is_spaces else request.url.scheme}://{request.url.hostname}{port}"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def landing_page():
    return HTMLResponse(f"""<p>ChatGPT Plugin for {tool}</p>""")


@app.get('/openapi.yaml', include_in_schema=False)
@functools.lru_cache()
def read_openapi_yaml() -> Response:
    openapi_json= app.openapi()
    yaml_s = io.StringIO()
    yaml.dump(openapi_json, yaml_s)
    return Response(yaml_s.getvalue(), media_type='text/yaml')


@app.get("/.well-known/ai-plugin.json")
@functools.lru_cache()
def ai_plugin(request: Request) -> JSONResponse:
    plugin = json.loads(json.dumps(plugin_json).replace("<<insert-url-here>>", get_url(request)))
    return JSONResponse(plugin)


@app.post("/predict")
def predict(body: PostBody, request: Request) -> JSONResponse:
    output = tool.run(body.query)
    if isinstance(output, str) and pathlib.Path(output).is_file():
        output = urllib.parse.urljoin(get_url(request), f"file={output}")
    return JSONResponse({"output": output})


@app.get("/file={path_or_url:path}")
async def file(path_or_url: str):

    path = pathlib.Path(path_or_url)

    if path.is_absolute():
        abs_path = path
    else:
        abs_path = path.resolve()

    if not abs_path.exists():
        raise HTTPException(404, "File not found")
    if abs_path.is_dir():
        raise HTTPException(403)
    return FileResponse(abs_path, headers={"Accept-Ranges": "bytes"})

uvicorn.run(app, host="0.0.0.0", port=7860)
