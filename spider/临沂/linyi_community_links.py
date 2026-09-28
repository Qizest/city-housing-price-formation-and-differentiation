import time
import os
import csv
import random
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# ================== 配置 ==================
CITY = "临沂"
CITY_PINYIN = "linyi"
BASE_URL = f"https://{CITY_PINYIN}.anjuke.com/community/p"
OUTPUT_DIR = os.path.join("..", "..", "data", CITY, "community")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"{CITY}小区_详情链接.csv")

EDGE_DRIVER_PATH = None
PAGE_LOAD_TIMEOUT = 45
SLEEP_RANGE = (2, 4)
MAX_PAGES = 50


# ================== 驱动管理 ==================
def get_edge_driver():
    options = Options()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--window-size=1920,1080')
    driver = webdriver.Edge(options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''Object.defineProperty(navigator, 'webdriver', { get: () => undefined })'''
    })
    return driver


# ================== 验证码处理 ==================
def check_for_captcha(driver):
    try:
        if driver.find_elements(By.CSS_SELECTOR, '.geetest_panel, .slide-verify, .captcha'):
            return True
        if any(k in driver.page_source for k in ['验证码', '滑动验证', '智能验证']):
            return True
        return False
    except:
        return False


def handle_captcha(driver):
    print("⚠️ 检测到验证码！请手动在浏览器中完成验证，然后按 Enter 继续...")
    input("按 Enter 继续...")
    driver.refresh()
    time.sleep(3)
    return not check_for_captcha(driver)


def safe_get(driver, url, retries=3):
    for attempt in range(retries):
        try:
            driver.get(url)
            time.sleep(2)
            if check_for_captcha(driver):
                if not handle_captcha(driver):
                    continue
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a.li-row'))
            )
            return True
        except Exception as e:
            print(f"⚠️ 加载页面失败 (尝试 {attempt+1}/{retries}): {e}")
            time.sleep(3)
    return False


# ================== 解析小区列表页（只提取链接 + 标签） ==================
def parse_community_page(soup):
    """
    提取：小区详情URL + 标签列表
    返回: list of dict
    """
    results = []
    items = soup.select('a.li-row')

    for a in items:
        try:
            # 小区详情页 URL
            href = a.get('href')
            if not href:
                continue
            if href.startswith('/'):
                href = 'https://' + CITY_PINYIN + '.anjuke.com' + href
            elif href.startswith('//'):
                href = 'https:' + href

            # 标签列表（多个 .prop-tag）
            tags = [tag.get_text(strip=True) for tag in a.select('.prop-tag')]
            tags_str = '|'.join(tags) if tags else ''

            results.append({
                '小区URL': href,
                '标签': tags_str
            })

        except Exception as e:
            print(f"⚠️ 解析单个小区出错: {e}")
            continue

    return results


# ================== 主程序 ==================
def main():
    print(f"🚀 启动 Edge 浏览器，爬取 {CITY} 小区列表（仅链接+标签）...")
    driver = get_edge_driver()

    all_results = []
    page = 1

    try:
        while page <= MAX_PAGES:
            url = f"{BASE_URL}{page}/"
            print(f"📄 正在爬取第 {page} 页: {url}")

            if not safe_get(driver, url):
                print(f"⚠️ 无法加载第 {page} 页，停止爬取")
                break

            html = driver.page_source
            soup = BeautifulSoup(html, 'lxml')
            results = parse_community_page(soup)

            if not results:
                print(f"📌 第 {page} 页没有小区，可能已到最后一页")
                break

            print(f"   ✅ 提取到 {len(results)} 个小区")
            all_results.extend(results)

            page += 1
            time.sleep(SLEEP_RANGE[0] + random.random() * (SLEEP_RANGE[1] - SLEEP_RANGE[0]))

    finally:
        driver.quit()

    # 去重（按URL）
    if all_results:
        seen = set()
        unique_results = []
        for r in all_results:
            if r['小区URL'] not in seen:
                seen.add(r['小区URL'])
                unique_results.append(r)

        print(f"📊 共获取到 {len(unique_results)} 个唯一小区")

        # 保存 CSV
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['序号', '小区URL', '标签'])
            for idx, r in enumerate(unique_results, 1):
                writer.writerow([idx, r['小区URL'], r['标签']])

        print(f"✅ 保存到 {OUTPUT_CSV}")
        print("\n📁 后续步骤:")
        print("   1. 运行 community_details 脚本从上述URL爬取详细信息")
        print("   2. 或直接用小区URL爬取房源链接")

    else:
        print("❌ 没有获取到任何小区")


if __name__ == "__main__":
    main()