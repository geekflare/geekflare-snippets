import base64
import os
import sys

from dotenv import load_dotenv
from geekflare_api.client import GeekflareClient
from geekflare_api.models import ScreenshotDto
from anthropic import Anthropic

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

geekflare_api_key = os.getenv("GEEKFLARE_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

url = "https://www.apple.com/iphone/"
SCREENSHOT_PATH = "screenshot.png"
REPORT_PATH = "qa_report.md"

# page_height caps the capture even with full_page=True, keeping the output
# under Claude's 8000px-per-dimension limit without client-side resizing.
with GeekflareClient(api_key=geekflare_api_key) as client:
    result = client.screenshot(
        ScreenshotDto(
            url=url,
            type="png",
            full_page=True,
            viewport_width=1280,
            page_height=5000,
            delay=5,
            inline=True,
        )
    )

image_data = result["inline"]["base64"]

with open(SCREENSHOT_PATH, "wb") as f:
    f.write(base64.b64decode(image_data))

claude = Anthropic(api_key=anthropic_api_key)

message = claude.messages.create(
    model="claude-opus-5",
    max_tokens=1500,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": """Analyze this webpage screenshot for visual QA.

Identify visible issues such as broken layouts, misplaced or overlapping
elements, missing content, incorrect spacing, image problems, navigation
issues, or other visual problems that could affect the user experience.

List each finding clearly. If you do not find any obvious visual issues,
state that.""",
                },
            ],
        }
    ],
)

report = next(block.text for block in message.content if block.type == "text")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(report)
