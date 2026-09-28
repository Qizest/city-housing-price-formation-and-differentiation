import time
import os
import random
import re
import csv
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup

# ================== 配置 ==================
INPUT_CSV = os.path.join("../..", "data", "金华", "rent", "金华租房_详情链接.csv")
OUTPUT_DIR = os.path.join("../..", "data", "金华", "rent")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, '金华租房_房源信息.csv')

MAX_HOUSES = None
MAX_RETRIES = 3
RETRY_DELAY = 5
PAGE_LOAD_TIMEOUT = 45
CONSECUTIVE_FAILURES_THRESHOLD = 5

EDGE_DRIVER_PATH = None

# ================== 驱动管理 ==================
def get_edge_driver():
    options = Options()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--window-size=1920,1080')
    options.page_load_strategy = 'normal'
    driver = webdriver.Edge(options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''Object.defineProperty(navigator, 'webdriver', { get: () => undefined })'''
    })
    return driver

def restart_driver(driver):
    try:
        driver.quit()
    except:
        pass
    time.sleep(2)
    print("🔄 正在重启浏览器驱动...")
    return get_edge_driver()

def check_for_captcha(driver):
    try:
        captcha_selectors = [
            '.geetest_panel', '.slide-verify', '.yidun_popup',
            '.captcha', '#captcha', '.nc_wrapper', '.gt_widget'
        ]
        for selector in captcha_selectors:
            if driver.find_elements(By.CSS_SELECTOR, selector):
                return True
        page_source = driver.page_source
        strict_keywords = ['请输入验证码', '滑动验证', '智能验证', '请完成验证', '点击验证']
        for keyword in strict_keywords:
            if keyword in page_source:
                return True
        if '安全验证' in page_source or (driver.title and '安全验证' in driver.title):
            return True
        return False
    except:
        return False

def handle_captcha(driver, url, max_attempts=3):
    if not check_for_captcha(driver):
        return True
    for attempt in range(max_attempts):
        print("⚠️ 检测到验证码！请手动在浏览器中完成验证，然后按 Enter 继续...")
        input("按 Enter 继续...")
        try:
            driver.refresh()
            time.sleep(3)
            if not check_for_captcha(driver):
                print("✅ 验证码已通过。")
                return True
            else:
                print(f"⚠️ 验证码仍然存在，剩余尝试次数：{max_attempts - attempt - 1}")
        except Exception as e:
            print(f"⚠️ 刷新页面时发生错误：{e}")
            time.sleep(RETRY_DELAY)
    print("❌ 多次尝试后验证码仍未通过，跳过该房源。")
    return False

def safe_fetch_url(driver, url, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            driver.get(url)
            time.sleep(2)
            if check_for_captcha(driver):
                if not handle_captcha(driver, url):
                    return False
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "house-title"))
            )
            return True
        except TimeoutException as e:
            if check_for_captcha(driver):
                if not handle_captcha(driver, url):
                    return False
                try:
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "house-title"))
                    )
                    return True
                except:
                    pass
            print(f"⚠️ 加载超时 (尝试 {attempt+1}/{retries}): {str(e)[:80]}")
            try:
                driver.execute_script("window.stop();")
            except:
                pass
            time.sleep(RETRY_DELAY * (attempt + 1))
        except (WebDriverException, ConnectionResetError) as e:
            print(f"⚠️ 驱动或网络错误 (尝试 {attempt+1}/{retries}): {str(e)[:80]}")
            time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            print(f"❌ 未知错误: {str(e)[:100]}")
            time.sleep(RETRY_DELAY)
    print(f"❌ 页面加载失败: {url[-60:]}")
    return False

def parse_detail_page(soup, url):
    data = {
        'house_id': re.search(r'/fangyuan/(\d+)', url).group(1) if re.search(r'/fangyuan/(\d+)', url) else None,
        'house_title': None,
        'district': None,
        'layout': None,
        'area': None,
        'total_price': None,
        'floor': None,
        'orientation': None,
        'decoration': None,
        'build_year': None,
        'elevator': None,
        'property_type': None,
        'page_tags': None,
    }

    title = soup.find('h1', class_='house-title')
    if title:
        data['house_title'] = title.get_text(strip=True)

    price_span = soup.find('span', class_='price')
    if price_span:
        match = re.search(r'(\d+)', price_span.get_text())
        if match:
            data['total_price'] = float(match.group(1))

    for li in soup.select('ul.house-info-zufang li, .house-info-detail li, .info-list li'):
        text = li.get_text(strip=True)
        if any(k in text for k in ['面积', '㎡']):
            m = re.search(r'(\d+\.?\d*)', text)
            if m:
                data['area'] = float(m.group(1))
        if any(k in text for k in ['室', '厅', '卫']) and not data['layout']:
            data['layout'] = text.split('：')[-1].strip() if '：' in text else text
        if '朝向' in text:
            data['orientation'] = text.split('：')[-1].strip()
        if '楼层' in text or '层' in text:
            data['floor'] = text.split('：')[-1].strip()
        if '装修' in text:
            data['decoration'] = text.split('：')[-1].strip()
        if any(k in text for k in ['类型', '物业类型', '住宅', '公寓']):
            data['property_type'] = text.split('：')[-1].strip()
        if '电梯' in text:
            data['elevator'] = text.split('：')[-1].strip()

    bread = soup.find('div', class_=re.compile('crumb|bread'))
    if bread:
        links = bread.find_all('a')
        if len(links) >= 3:
            for link in reversed(links):
                text = link.get_text(strip=True)
                if '区' in text and '金华' not in text:
                    data['district'] = text
                    break
    if not data['district']:
        for li in soup.select('ul.house-info-zufang li.house-info-item'):
            label = li.find('span', class_='type')
            if label and '小区' in label.get_text():
                district_links = li.find_all('a', href=re.compile(r'/fangyuan/'))
                if len(district_links) >= 1:
                    data['district'] = district_links[0].get_text(strip=True)
                    break

    general = soup.find('div', class_='auto-general')
    if general:
        data['page_tags'] = general.get_text(strip=True)

    return data

def main():
    if not os.path.exists(INPUT_CSV):
        print("❌ 未找到链接文件！")
        return

    df_links = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    if '状态' not in df_links.columns:
        df_links['状态'] = ''
    else:
        df_links['状态'] = df_links['状态'].fillna('').astype(str)

    mask = (df_links['状态'].isna()) | (df_links['状态'] == '') | (df_links['状态'] == '失败')
    df_todo = df_links[mask].copy()
    if MAX_HOUSES and MAX_HOUSES < len(df_todo):
        df_todo = df_todo.iloc[:MAX_HOUSES]

    total_to_crawl = len(df_todo)
    if total_to_crawl == 0:
        print("✅ 所有链接均已爬取，无需操作。")
        return

    print(f"📄 共有 {total_to_crawl} 条待爬取链接（状态为空或失败）")

    existing_df = pd.DataFrame()
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV, encoding='utf-8-sig')
        print(f"📂 已读取 {len(existing_df)} 条已有房源数据")

    driver = get_edge_driver()
    new_results = []
    total_success = 0
    total_failed = 0
    consecutive_failures = 0

    print(f"🚀 开始爬取详情页（共 {total_to_crawl} 条）...")

    for idx_in_df in df_todo.index:
        row = df_todo.loc[idx_in_df]
        url = row['详情页URL']

        print(f"[{total_success + total_failed + 1}/{total_to_crawl}] 正在爬取: {url}")

        if not safe_fetch_url(driver, url):
            total_failed += 1
            consecutive_failures += 1
            df_links.at[idx_in_df, '状态'] = '失败'
            if consecutive_failures >= CONSECUTIVE_FAILURES_THRESHOLD:
                print(f"⚠️ 连续失败 {consecutive_failures} 次，重启浏览器驱动...")
                driver = restart_driver(driver)
                consecutive_failures = 0
            if (total_success + total_failed) % 10 == 0:
                df_links.to_csv(INPUT_CSV, index=False, encoding='utf-8-sig')
            continue
        else:
            consecutive_failures = 0

        try:
            soup = BeautifulSoup(driver.page_source, 'lxml')
            detail = parse_detail_page(soup, url)
            if detail and detail.get('house_id'):
                combined = {
                    'city': '金华',
                    'house_type': '租房',
                    'community_name': row.get('小区名称', ''),
                    'community_id': row.get('小区ID', ''),
                    'tags': row.get('tags', ''),
                    'house_id': detail.get('house_id'),
                    'house_title': detail.get('house_title'),
                    'district': detail.get('district'),
                    'layout': detail.get('layout'),
                    'area': detail.get('area'),
                    'total_price': detail.get('total_price'),
                    'floor': detail.get('floor'),
                    'orientation': detail.get('orientation'),
                    'decoration': detail.get('decoration'),
                    'build_year': detail.get('build_year'),
                    'elevator': detail.get('elevator'),
                    'property_type': detail.get('property_type'),
                    'page_tags': detail.get('page_tags'),
                }
                new_results.append(combined)
                total_success += 1
                df_links.at[idx_in_df, '状态'] = '已爬取'
            else:
                print(f"   ❌ 解析结果为空")
                total_failed += 1
                df_links.at[idx_in_df, '状态'] = '失败'
        except Exception as e:
            print(f"   ❌ 解析异常: {str(e)[:80]}")
            total_failed += 1
            df_links.at[idx_in_df, '状态'] = '失败'

        if (total_success + total_failed) % 10 == 0:
            df_links.to_csv(INPUT_CSV, index=False, encoding='utf-8-sig')
            if new_results:
                combined = pd.concat([existing_df, pd.DataFrame(new_results)], ignore_index=True)
                if 'house_id' in combined.columns:
                    combined['house_id'] = combined['house_id'].astype(str)
                combined.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
                existing_df = combined
                print(f"   💾 已保存 {len(combined)} 条房源数据（本次新增 {len(new_results)} 条）")
                new_results = []

        time.sleep(3 + random.random() * 3)

    driver.quit()

    df_links.to_csv(INPUT_CSV, index=False, encoding='utf-8-sig')
    if new_results:
        combined = pd.concat([existing_df, pd.DataFrame(new_results)], ignore_index=True)
        if 'house_id' in combined.columns:
            combined['house_id'] = combined['house_id'].astype(str)
        combined.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\n✅ 爬取完成！共 {len(combined)} 条房源数据 → {OUTPUT_CSV}")
    else:
        print(f"\n✅ 没有新数据需要保存，已有 {len(existing_df)} 条房源数据")

    print(f"📊 本次成功爬取 {total_success} 条，失败 {total_failed} 条")

    status_counts = df_links['状态'].value_counts()
    print("\n📊 链接文件状态统计:")
    for status, count in status_counts.items():
        print(f"   {status}: {count}")

if __name__ == "__main__":
    main()