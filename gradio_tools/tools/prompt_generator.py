from __future__ import annotations

from gradio_client.client import Job

from gradio_tools.tools.gradio_tool import GradioTool


class StableDiffusionPromptGeneratorTool(GradioTool):
    def __init__(
        self,
        name="StableDiffusionPromptGenerator",
        description=(
            "Use this tool to improve a prompt for stable diffusion and other image generators "
            "This tool will refine your prompt to include key words and phrases that make "
            "stable diffusion perform better. The input is a prompt text string "
            "and the output is a prompt text string"
        ),
        src="microsoft/Promptist",
        hf_token=None,
    ) -> None:
        super().__init__(name, description, src, hf_token)

    def create_job(self, query: str) -> Job:
        return self.client.submit(query, api_name="/predict")

    def postprocess(self, output: str) -> str:
        return output
