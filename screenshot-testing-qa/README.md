# Screenshot Testing QA

Capture a webpage screenshot with the [Geekflare Screenshot API](https://docs.geekflare.com/endpoint/screenshot) and have Claude analyze it for visual QA issues — broken layouts, missing content, overlapping elements, low-contrast text, and other UX problems.

## How it works

1. Requests a screenshot from Geekflare (`geekflare-api` Python SDK), returned inline as base64 so no extra download step is needed.
2. Saves the screenshot to `screenshot.png`.
3. Sends the image to Claude (Anthropic API) with a visual QA prompt.
4. Saves Claude's findings to `qa_report.md` and prints them to stdout.

## Prerequisites

- Python 3.10+
- A [Geekflare API key](https://geekflare.com/api/)
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in this directory:

```
GEEKFLARE_API_KEY="your_geekflare_api_key"
ANTHROPIC_API_KEY="your_anthropic_api_key"
```

## Usage

```bash
python screenshot_testing_qa.py
```

Edit the `url` variable in `screenshot_testing_qa.py` to point at the page you want to test.

## Notes

- `page_height=5000` caps the screenshot height even with `full_page=True`, keeping the image under Claude's 8000px-per-dimension limit without any client-side resizing.
- `delay=5` waits 5 seconds before capture so scroll-triggered animations and lazy-loaded images finish rendering — without it, fade-in transitions can be caught mid-animation and misread as rendering bugs.
- `screenshot.png` and `qa_report.md` are regenerated on every run.
