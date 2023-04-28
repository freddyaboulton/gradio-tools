from typing import TYPE_CHECKING
from urllib.parse import urljoin

if TYPE_CHECKING:
    from gradio_tools.tools import GradioTool

def make_manifest(tool: 'GradioTool', version: str, private: bool, email: str):

    auth = {
        "type": "user_http",
        "authorization_type": "bearer",
    } if private else {"type": "none"}

    return {
        "schema_version": version,
        "name_for_human": tool.name,
        "name_for_model": tool.name,
        "description_for_human": f"{tool.name}: a plugin based for the gradio space hosted on {tool.src}",
        "description_for_model": tool.description,
        "auth": auth,
        "api": {
            "type": "openapi",
            "url": "<<insert-url-here>>/openapi.json",
            "is_user_authenticated": False
        },
        "logo_url": "<<insert-url-here>>/favicon.ico",
        "contact_email": email,
        "legal_info_url": "<<insert-url-here>>"
    }