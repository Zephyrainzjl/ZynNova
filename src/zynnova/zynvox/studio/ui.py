"""Optional Gradio UI backed by the same ZynVoxStudio Python API."""
from __future__ import annotations

from .studio import ZynVoxStudio
from .types import GenerationRequest


def build_ui(studio: ZynVoxStudio | None = None):
    try: import gradio as gr
    except ImportError as exc: raise RuntimeError("install zynnova[voice-ui]") from exc
    studio=studio or ZynVoxStudio()
    def run(text,voice,language,speed):
        return str(studio.synthesize(GenerationRequest(text=text,voice_id=voice,language=language,speed=float(speed))).audio)
    with gr.Blocks(title="ZynVox Studio") as demo:
        gr.Markdown("# ZynVox Studio")
        text=gr.Textbox(label="Text",lines=5); voice=gr.Dropdown(choices=list(studio.list_voices()),label="Voice")
        language=gr.Textbox(value="auto",label="Language"); speed=gr.Slider(0.5,2.0,value=1.0,label="Speed")
        output=gr.Audio(label="Output",type="filepath"); gr.Button("Synthesize").click(run,[text,voice,language,speed],output)
    return demo


__all__=["build_ui"]
