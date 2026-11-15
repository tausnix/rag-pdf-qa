"""Capture the hero screenshot from the deployed Streamlit Cloud app.

Streamlit Community Cloud wraps the app in a same-origin iframe (`/~/+/`), so
every locator has to be resolved inside that frame -- the outer document is
just Streamlit's toolbar shell and has no app text at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/tracerag-hero.png")
URL = "https://tracerag.streamlit.app"
PRESET = "Rank rescue · Meridian headcount"
TMP = Path("C:/Users/SEEMA/AppData/Local/Temp/opencode")


def app_frame(page, timeout_ms: int = 240_000):
    """Return the frame that actually hosts the Streamlit app."""
    waited = 0
    while waited < timeout_ms:
        for frame in page.frames:
            if "/~/+/" in frame.url:
                return frame
        page.wait_for_timeout(1000)
        waited += 1000
    raise SystemExit("the app iframe never appeared")


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1400}, device_scale_factor=2)
    page.goto(URL, wait_until="load", timeout=180_000)

    frame = app_frame(page)

    # app boot (a cold Streamlit Cloud container indexes the corpus first)
    frame.get_by_text("One-click demos").first.wait_for(timeout=240_000)
    text = frame.inner_text("body")
    if "Indexing failed" in text:
        raise SystemExit("sidebar shows an indexing error:\n" + text[:1200])

    frame.get_by_text(PRESET, exact=True).first.click(timeout=60_000)

    frame.get_by_text("Candidate ranking").first.wait_for(timeout=300_000)
    frame.get_by_text("CE Confidence").first.wait_for(timeout=120_000)
    page.wait_for_timeout(3000)

    text = frame.inner_text("body")
    if "Indexing failed" in text:
        raise SystemExit("sidebar reported an indexing error after the query")
    print("gemini answer:", "[extractive fallback" not in text)
    print("side error:", "Indexing failed" in text)
    print("has 44 employees:", "44" in text)
    print("movement pill:", "Rerank #1" in text)

    iframe_box = page.locator("iframe[src*='/~/+/']").bounding_box()
    print("iframe box:", iframe_box)
    print("outer doc height:", page.evaluate("document.documentElement.scrollHeight"))

    page.screenshot(path=str(TMP / "hero_outer.png"))
    page.locator("iframe[src*='/~/+/']").screenshot(path=str(TMP / "hero_iframe.png"))
    browser.close()

print("wrote", TMP / "hero_outer.png", "and", TMP / "hero_iframe.png")
print("target:", OUT)
