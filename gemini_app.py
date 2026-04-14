import os
import sys
import asyncio
import random
from playwright.async_api import async_playwright

AUTH_FILE = "auth.json"
USER_DATA_DIR = "./playwright_user_data"

async def get_gemini_response(prompt_text):
    async with async_playwright() as p:
        browser_type = p.chromium
        
        context = await browser_type.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=False,
            viewport={'width': 1280, 'height': 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )

        page = await context.new_page()
        await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        print(f"Opening Gemini...")
        try:
            # Dùng domcontentloaded để vào việc nhanh hơn
            await page.goto("https://gemini.google.com/app?hl=en", wait_until="domcontentloaded")
        except:
            pass

        prompt_selector = 'div[aria-label="Enter a prompt for Gemini"], [contenteditable="true"]'

        try:
            # Đợi ô nhập liệu xuất hiện thẳng luôn
            await page.wait_for_selector(prompt_selector, timeout=20000)
            await page.click(prompt_selector)
            await page.fill(prompt_selector, prompt_text)
            
            print("--- Sending Prompt ---")
            await page.keyboard.press("Enter")

            # Đợi phản hồi hoàn tất
            stop_btn = 'button[aria-label="Stop response"]'
            send_btn = 'button[aria-label="Send message"]'
            
            for _ in range(30):
                await page.wait_for_timeout(1000)
                is_generating = await page.query_selector(stop_btn)
                btn_ready = await page.query_selector(send_btn)
                if not is_generating and btn_ready:
                    if await btn_ready.is_enabled():
                        break
            
            await page.wait_for_timeout(500)

            # Lấy kết quả
            response_selector = 'message-content'
            responses = await page.query_selector_all(response_selector)
            if responses:
                response_text = await responses[-1].inner_text()
                print("\n" + "="*50 + "\nDONE\n" + "="*50)
                with open("ans.txt", "w", encoding="utf-8") as f:
                    f.write(response_text)
                await context.storage_state(path=AUTH_FILE)
                await context.close()
                return response_text
        except Exception as e:
            print(f"Error during prompt: {e}")
            
        await context.close()
        return None

if __name__ == "__main__":
    asyncio.run(get_gemini_response("Ping!"))
