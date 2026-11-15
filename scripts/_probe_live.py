"""Why do text locators miss? Inspect frame/shadow structure of the live app."""
from __future__ import annotations

from playwright.sync_api import sync_playwright

URL = "https://tracerag.streamlit.app"

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto(URL, wait_until="load", timeout=120_000)
    page.wait_for_timeout(20_000)

    print("frames:", len(page.frames))
    for f in page.frames:
        try:
            txt = f.inner_text("body")[:120]
        except Exception as exc:  # noqa: BLE001
            txt = f"<{type(exc).__name__}: {exc}>"
        print("  frame:", repr(f.url[:90]), "|", repr(txt))

    print("main body len:", len(page.main_frame.inner_text("body")))
    print("element count:", page.evaluate("document.querySelectorAll('*').length"))
    print("shadow roots:", page.evaluate(
        "Array.from(document.querySelectorAll('*')).filter(e => e.shadowRoot).map(e => e.tagName)"
    ))
    print("body innerText len:", page.evaluate("document.body.innerText.length"))
    print("textContent has demos:", "One-click demos" in page.evaluate("document.body.textContent"))
    print("preset count:", page.get_by_text("Rank rescue", exact=False).count())
    print("demos count:", page.get_by_text("One-click demos", exact=False).count())
    print("buttons:", [b[:40] for b in page.locator("button").all_inner_texts()][:12])
    browser.close()
