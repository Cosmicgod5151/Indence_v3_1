"""Capture report screenshots from the running app (backend :8000, frontend :3000).

Uses the installed Microsoft Edge through Playwright (no browser download):
    python report\\take_screenshots.py
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
URL = "http://localhost:3000"
LONG = 240_000


def shot(page, selector, name, nth=0):
    try:
        loc = page.locator(selector).nth(nth)
        loc.scroll_into_view_if_needed(timeout=5000)
        loc.screenshot(path=str(OUT / name), timeout=15000)
        print("saved", name)
    except Exception as exc:  # keep going; report what is missing
        print("MISSING", name, type(exc).__name__, str(exc)[:120])


def wait_idle(page, button_text):
    page.wait_for_function(
        """t => [...document.querySelectorAll('button.search-btn')].some(b => b.textContent.trim() === t && !b.disabled)""",
        arg=button_text, timeout=LONG)
    page.wait_for_timeout(1500)


def run_search(page, q):
    page.locator("input[placeholder^='Enter clinical query']").fill(q)
    page.get_by_role("button", name="Synthesize Evidence").click()
    page.wait_for_timeout(1000)
    wait_idle(page, "Synthesize Evidence")


def main(which):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
        page.goto(URL)
        page.wait_for_load_state("networkidle")
        if page.locator("input[type=password]").count():
            page.screenshot(path=str(OUT / "ss_login.png"))
            page.get_by_role("button", name="Sign In to Portal").click()
            page.wait_for_selector(".nav-tabs", timeout=30000)
        page.add_style_tag(content=".app-header{position:static !important;}")

        if "search" in which:
            run_search(page, "Is trastuzumab deruxtecan effective in patients with HER2-positive breast cancer?")
            page.screenshot(path=str(OUT / "ss_search_top.png"))
            shot(page, ".synthesis-container", "ss_answer.png")
            shot(page, ".model-trace", "ss_model_trace.png")
            shot(page, ".certainty-card", "ss_certainty.png")
            shot(page, ".claim-verification", "ss_claims.png")
            run_search(page, "Does adding immunotherapy to chemotherapy improve overall survival in advanced non-small cell lung cancer?")
            shot(page, ".evidence-map", "ss_forest.png")
            shot(page, ".certainty-card", "ss_certainty2.png")
            page.get_by_role("button", name="Execution Audit").click()
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / "ss_audit.png"), full_page=False)
            page.get_by_role("button", name="Clinical Search").click()

        if "abstain" in which:
            run_search(page, "Is zorvatinib effective for EGFR-mutant non-small cell lung cancer?")
            page.screenshot(path=str(OUT / "ss_abstain_entity.png"))

        if "case" in which:
            page.get_by_role("button", name="Patient Case").click()
            page.get_by_role("button", name="Privacy demo (blocked)").click()
            page.locator(".case-panel button.search-btn").first.click()
            page.wait_for_timeout(2500)
            page.screenshot(path=str(OUT / "ss_case_phi.png"))

            page.get_by_role("button", name="NSCLC · conflicting PD-L1").click()
            btn = page.locator(".case-panel button.search-btn").first
            btn.click()
            page.wait_for_selector(".case-conflict, .error-box", timeout=LONG)
            page.wait_for_timeout(1500)
            shot(page, ".case-grid", "ss_case_facts.png")

            page.get_by_role("button", name="Breast · HR+ after CDK4/6i").click()
            page.locator(".case-panel button.search-btn").first.click()
            page.wait_for_function(
                "() => ![...document.querySelectorAll('button')].some(b => b.textContent.includes('Screening & extracting'))",
                timeout=LONG)
            page.wait_for_timeout(1500)
            shot(page, ".case-grid", "ss_case_facts_hr.png")
            page.get_by_role("button", name="Find applicable evidence").click()
            page.wait_for_timeout(1500)
            page.wait_for_function(
                "() => ![...document.querySelectorAll('button')].some(b => b.textContent.includes('Searching & appraising'))",
                timeout=LONG)
            page.wait_for_timeout(2000)
            shot(page, ".case-results .case-panel", "ss_case_answer.png", 0)
            shot(page, ".safety-check", "ss_safety.png")
            shot(page, ".case-panel:has-text('Applicability Matrix')", "ss_applicability.png")
            page.screenshot(path=str(OUT / "ss_case_full.png"), full_page=True)
        browser.close()


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {"search", "abstain", "case"})
