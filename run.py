from fastapi import FastAPI
import gradio as gr

from app import app_ui

app = FastAPI()

@app.get('/')
async def root():
    return 'Gradio app is running at /gradio', 200

app = gr.mount_gradio_app(app, app_ui, path='/gradio')