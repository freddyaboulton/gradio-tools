from minichain import show, prompt, OpenAI, OpenAIStream
import gradio as gr
from gradio_tools.tools import StableDiffusionTool, ImageCaptioningTool

@prompt(OpenAIStream(), stream=True)
def picture(model, query):
    out = ""
    for r in model.stream(query):
        out += r
        yield out

@prompt(StableDiffusionTool(), stream=True, block_input=lambda: gr.Textbox(label=""))
def gen(model, query):
    for r in model.stream(query):
        # Show a blue cube while waiting.
        yield "https://htmlcolorcodes.com/assets/images/colors/baby-blue-color-solid-background-1920x1080.png"
    yield r

@prompt(ImageCaptioningTool(), block_output=lambda: gr.Textbox(label=""))
def caption(model, img_src):
    return model(img_src)

def gradio_example(query):
    return caption(gen(picture(query)))


desc = """
### Gradio Tool

Examples using the gradio tool in Minichain. Example creates an image and then captions it.

"""


gradio = show(gradio_example,
              subprompts=[picture, gen, caption],
              examples=['Describe a one-sentence fantasy scene.',
                        'Describe a one-sentence scene happening on the moon.'],
              out_type="markdown",
              description=desc,
              css="#advanced {display: none}"

              )
if __name__ == "__main__":
    gradio.queue().launch()

