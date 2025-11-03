from fastapi import FastAPI
from nomad.config import config

chatbot_api_entry_point = config.get_plugin_entry_point('nomad_compass.apis:chatbot_api')

app = FastAPI(
    root_path=f'{config.services.api_base_path}/{chatbot_api_entry_point.prefix}'
)

@app.get('/')
async def root():
    return {"message": "Hello World"}