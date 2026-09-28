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
INPUT_CSV = os.path.join("../..", "data", "临沂", "sale", "临沂二手房_详情链接.csv")
OUTPUT_DIR = os.path.join("../..", "data", "临沂", "sale")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, '临沂二手房_房源信息.csv')

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

# ================== 验证码检测（增强版） ==================
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
                EC.presence_of_element_located((By.CSS_SELECTOR, 'h1.title'))
            )
            return True
        except TimeoutException as e:
            if check_for_captcha(driver):
                if not handle_captcha(driver, url):
                    return False
                try:
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'h1.title'))
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

# ================== 解析详情页 ==================
def parse_detail_page(soup, url):
    data = {
        'house_id': None,
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

    match = re.search(r'/prop/view/([A-Za-z0-9]+)', url) or re.search(r'/sale/(\d+)', url)
    if match:
        data['house_id'] = match.group(1)

    title_tag = soup.select_one('h1.title')
    if title_tag:
        data['house_title'] = title_tag.get_text(strip=True)

    price_num = soup.select_one('.maininfo-price-num')
    if price_num:
        price_text = price_num.get_text(strip=True)
        match_price = re.search(r'(\d+\.?\d*)', price_text)
        if match_price:
            data['total_price'] = match_price.group(1)

    model = soup.select_one('.maininfo-model')
    if model:
        items = model.select('.maininfo-model-item')
        for item in items:
            strong = item.select_one('.maininfo-model-strong')
            weak = item.select_one('.maininfo-model-weak')
            if strong:
                strong_text = strong.get_text(strip=True)
                if re.search(r'\d+室\d+厅', strong_text):
                    match_layout = re.search(r'(\d+室\d+厅)', strong_text)
                    if match_layout:
                        data['layout'] = match_layout.group(1)
                elif '㎡' in strong_text or 'm²' in strong_text:
                    match_area = re.search(r'(\d+\.?\d*)\s*[㎡m²]', strong_text)
                    if match_area:
                        data['area'] = match_area.group(1)
                else:
                    i_tag = strong.find('i')
                    if i_tag:
                        orientation_text = i_tag.get_text(strip=True)
                        if orientation_text:
                            data['orientation'] = orientation_text
                    else:
                        if not re.search(r'\d', strong_text) and strong_text.strip():
                            data['orientation'] = strong_text.strip()
            if weak:
                weak_text = weak.get_text(strip=True)
                if '层' in weak_text and not data['floor']:
                    data['floor'] = weak_text
                if '年' in weak_text and not data['build_year']:
                    year_match = re.search(r'(\d{4})年', weak_text)
                    if year_match:
                        data['build_year'] = year_match.group(1)
                if not data['decoration']:
                    decoration_keywords = ['毛坯', '精装修', '简装', '豪华装修', '普通装修', '简单装修']
                    for kw in decoration_keywords:
                        if kw in weak_text:
                            data['decoration'] = kw
                            break
                if not data['property_type']:
                    property_keywords = ['公寓', '普通住宅', '别墅', '平房', '排屋', '其他']
                    for kw in property_keywords:
                        if kw in weak_text:
                            data['property_type'] = kw
                            break

    community_items = soup.select('div.maininfo-community-item')
    for item in community_items:
        label_span = item.find('span', class_='maininfo-community-item-label')
        if not label_span:
            continue
        label_text = label_span.get_text(strip=True)
        if label_text == '所属区域':
            area_links = item.find_all('a', href=re.compile(r'/sale/'))
            district = None
            for link in area_links:
                text = link.get_text(strip=True)
                if '二手房' in text and '临沂' not in text:
                    district = text.replace('二手房', '').strip()
                    break
            if not district and area_links:
                district = area_links[0].get_text(strip=True).replace('二手房', '').strip()
            data['district'] = district
            break

    if not data['district']:
        crumbs = soup.select('.crumbs a')
        if len(crumbs) >= 3:
            for crumb in crumbs:
                text = crumb.get_text(strip=True)
                if '二手房' in text and '临沂' not in text:
                    data['district'] = text.replace('二手房', '').strip()
                    break

    tags_div = soup.find('div', class_='maininfo-tags')
    if tags_div:
        tag_spans = tags_div.find_all('span', class_='maininfo-tags-item')
        for span in tag_spans:
            if '有电梯' in span.get_text(strip=True):
                data['elevator'] = '有电梯'
                break

    intro_content = soup.select_one('.houseIntro-content')
    if intro_content:
        data['page_tags'] = intro_content.get_text(strip=True)

    return data

# ================== 主程序 ==================
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
    total_skipped = 0
    consecutive_failures = 0

    print(f"🚀 开始爬取详情页（共 {total_to_crawl} 条）...")

    for idx_in_df in df_todo.index:
        row = df_todo.loc[idx_in_df]
        url = row['详情页URL']

        if '/xinfang/' in url:
            print(f"[{total_success + total_failed + total_skipped + 1}/{total_to_crawl}] 🏠 检测到新房链接，跳过: {url}")
            total_skipped += 1
            df_links.at[idx_in_df, '状态'] = '新房跳过'
            if (total_success + total_failed + total_skipped) % 10 == 0:
                df_links.to_csv(INPUT_CSV, index=False, encoding='utf-8-sig')
            continue

        print(f"[{total_success + total_failed + total_skipped + 1}/{total_to_crawl}] 正在爬取: {url}")

        if not safe_fetch_url(driver, url):
            total_failed += 1
            consecutive_failures += 1
            df_links.at[idx_in_df, '状态'] = '失败'
            if consecutive_failures >= CONSECUTIVE_FAILURES_THRESHOLD:
                print(f"⚠️ 连续失败 {consecutive_failures} 次，重启浏览器驱动...")
                driver = restart_driver(driver)
                consecutive_failures = 0
            if (total_success + total_failed + total_skipped) % 10 == 0:
                df_links.to_csv(INPUT_CSV, index=False, encoding='utf-8-sig')
            continue
        else:
            consecutive_failures = 0

        try:
            soup = BeautifulSoup(driver.page_source, 'lxml')
            detail = parse_detail_page(soup, url)
            if detail and detail.get('house_id'):
                combined = {
                    'city': '临沂',
                    'house_type': '二手房',
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

        if (total_success + total_failed + total_skipped) % 10 == 0:
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

    print(f"📊 本次成功爬取 {total_success} 条，失败 {total_failed} 条，跳过新房 {total_skipped} 条")

    status_counts = df_links['状态'].value_counts()
    print("\n📊 链接文件状态统计:")
    for status, count in status_counts.items():
        print(f"   {status}: {count}")

if __name__ == "__main__":
    main()