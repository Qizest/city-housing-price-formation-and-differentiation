import time
import os
import csv
import re
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# ================== 配置 ==================
CITY = "金华"
CITY_PINYIN = "jinhua"
MAX_LINKS_PER_PAGE = 10
SLEEP_RANGE = (3, 6)  # 每个小区之间随机延时

COMMUNITY_LINKS_CSV = os.path.join("..", "..", "data", CITY, "community", f"{CITY}小区_详情链接.csv")
COMMUNITY_INFO_CSV = os.path.join("..", "..", "data", CITY, "community", f"{CITY}小区信息.csv")
SALE_LINKS_CSV = os.path.join("..", "..", "data", CITY, "sale", f"{CITY}二手房_详情链接.csv")
RENT_LINKS_CSV = os.path.join("..", "..", "data", CITY, "rent", f"{CITY}租房_详情链接.csv")

os.makedirs(os.path.dirname(SALE_LINKS_CSV), exist_ok=True)
os.makedirs(os.path.dirname(RENT_LINKS_CSV), exist_ok=True)

EDGE_DRIVER_PATH = None
PAGE_LOAD_TIMEOUT = 90


# ================== 驱动管理 ==================
def get_edge_driver():
    options = Options()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument(
        '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.page_load_strategy = 'eager'

    driver = webdriver.Edge(options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        '''
    })
    return driver


# ================== 验证码处理 ==================
def check_for_captcha(driver):
    try:
        if driver.find_elements(By.CSS_SELECTOR, '.geetest_panel, .slide-verify, .captcha'):
            return True
        if any(k in driver.page_source for k in ['验证码', '滑动验证', '智能验证']):
            return True
        # 检测页面标题或内容是否包含“安全验证”
        if driver.title and '安全验证' in driver.title:
            return True
        if '安全验证' in driver.page_source:
            return True
        return False
    except:
        return False


def handle_captcha(driver, max_attempts=3):
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
            time.sleep(3)
    print("❌ 多次尝试后验证码仍未通过")
    return False


# ================== 安全加载页面 ==================
def safe_fetch(driver, url, house_type, retries=3):
    selectors = {
        'sale': ['.property', '.list-box', '.no-data'],
        'rent': ['.rent-link', '.list-box', '.no-data']
    }
    wait_selectors = selectors.get(house_type, selectors['sale'])

    for attempt in range(retries):
        try:
            driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
            driver.get(url)
            time.sleep(3)

            if check_for_captcha(driver):
                if not handle_captcha(driver):
                    continue

            found = False
            for selector in wait_selectors:
                try:
                    WebDriverWait(driver, 8).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    found = True
                    break
                except:
                    continue
            if not found:
                time.sleep(2)

            time.sleep(1)
            return True

        except Exception as e:
            print(f"⚠️ 加载 {house_type} 页面失败 (尝试 {attempt + 1}/{retries}): {e}")
            try:
                driver.execute_script("window.stop();")
            except:
                pass
            time.sleep(5)
    return False


# ================== 提取二手房 tags ==================
def extract_sale_tags(card):
    tags = []

    # 1. 从 property-content-title-othertag 提取
    tag_container = card.select_one('.property-content-title-othertag')
    if tag_container:
        for tag_el in tag_container.select('.tag, .item-tag, span, a'):
            t = tag_el.get_text(strip=True)
            if t and len(t) <= 15 and t not in tags:
                tags.append(t)

    # 2. 如果 othertag 为空，从标题中提取关键词
    if not tags:
        title_el = card.select_one('.property-content-title-name')
        if title_el:
            title_text = title_el.get_text(strip=True)
            keywords = ['满2年', '满5年', '满二', '满五', '唯一',
                        '精装修', '毛坯', '简装', '豪装', '中等装修',
                        '高楼层', '低楼层', '中层', '低层', '高层', '顶楼', '底楼',
                        '电梯房', '无电梯', '步梯',
                        '南北通透', '朝南', '朝北', '朝东', '朝西', '全明户型',
                        '拎包入住', '随时看房', '近地铁', '近学校', '近商场',
                        '视野开阔', '采光好', '户型方正', '双卫', '双阳台',
                        '降价', '急售', '首付低', '得房率高', '带车位',
                        '一线山景', '湖景', '江景', '园景', '独栋', '联排', '双拼']
            for kw in keywords:
                if kw in title_text and kw not in tags:
                    tags.append(kw)

    # 3. 检测安选标签
    anxuan_img = card.select_one('.property-content-title-anxuan')
    if anxuan_img:
        tags.append('安选')

    tags = list(dict.fromkeys(tags))
    return '|'.join(tags[:5]) if tags else ''


# ================== 提取第一页房源链接 + tags ==================
def extract_first_page_links(driver, url, house_type, community_id, community_name, max_links=10):
    if not safe_fetch(driver, url, house_type):
        print(f"      ⚠️ 页面加载失败，跳过")
        return []

    soup = BeautifulSoup(driver.page_source, 'lxml')
    results = []

    if house_type == 'sale':
        items = soup.select('div.property')
        if not items:
            items = soup.select('a.property-ex')
            for a in items:
                href = a.get('href')
                if href and '/prop/view/' in href:
                    full_url = normalize_url(href, 'sale')
                    if full_url:
                        results.append({'url': full_url, 'tags': ''})
        else:
            for card in items:
                a_tag = card.select_one('a.property-ex')
                if not a_tag:
                    a_tag = card.find('a', href=True)
                if a_tag:
                    href = a_tag.get('href')
                    if href and '/prop/view/' in href:
                        full_url = normalize_url(href, 'sale')
                        if full_url:
                            tags_str = extract_sale_tags(card)
                            results.append({'url': full_url, 'tags': tags_str})

    else:
        # 租房
        items = soup.select('a.rent-link')
        for a in items:
            href = a.get('href')
            if href and ('/fangyuan/' in href or '/rent/' in href):
                full_url = normalize_url(href, 'rent')
                if full_url:
                    tags = []
                    tag_container = a.select_one('.rent-arch')
                    if tag_container:
                        for tag in tag_container.select('.item-tag'):
                            t = tag.get_text(strip=True)
                            if t:
                                tags.append(t)
                    tags_str = '|'.join(tags) if tags else ''
                    results.append({'url': full_url, 'tags': tags_str})

    if not results:
        no_data = soup.find(string=re.compile('暂无房源|没有房源|0套'))
        if no_data:
            print(f"      📌 该小区没有 {house_type} 房源")
        else:
            print(f"      📌 未找到 {house_type} 房源链接")
            title = soup.find('title')
            if title:
                print(f"         页面标题: {title.get_text(strip=True)}")
        return []

    seen = set()
    unique_results = []
    for item in results:
        if item['url'] not in seen:
            seen.add(item['url'])
            unique_results.append(item)

    if len(unique_results) > max_links:
        unique_results = unique_results[:max_links]
    print(f"      ✅ 提取到 {len(unique_results)} 条 {house_type} 链接（限制{max_links}条）")
    return unique_results


def normalize_url(href, house_type):
    if not href:
        return None
    href = href.strip()
    if href.startswith('//'):
        return 'https:' + href
    elif href.startswith('/'):
        if house_type == 'sale':
            return f'https://{CITY_PINYIN}.anjuke.com' + href
        else:
            return f'https://{CITY_PINYIN}.anjuke.com' + href
    elif href.startswith('http'):
        return href.split('?')[0]
    else:
        return href


# ================== 保存链接（去重追加） ==================
def save_links_to_csv(new_links, csv_path, label, community_id, community_name):
    """
    新链接写入 CSV，自动去重。
    new_links: list of dict {'url': ..., 'tags': ...}
    """
    if not new_links:
        return

    # 读取已有链接，建立集合用于去重
    existing_links = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            if '详情页URL' in df.columns:
                existing_links = set(df['详情页URL'].dropna().tolist())
        except:
            pass

    # 过滤掉已存在的链接
    filtered_links = []
    for item in new_links:
        if item['url'] not in existing_links:
            filtered_links.append(item)

    if not filtered_links:
        print(f"   📌 {label}: 无新链接（全部已存在）")
        return

    # 追加写入
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if os.path.getsize(csv_path) == 0:
            writer.writerow(['序号', '小区名称', '小区ID', '详情页URL', 'tags'])

        # 计算起始序号
        start_idx = 1
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8-sig')
                if not df.empty and '序号' in df.columns:
                    start_idx = df['序号'].max() + 1
            except:
                pass

        for idx, item in enumerate(filtered_links, start=start_idx):
            writer.writerow([idx, community_name, community_id, item['url'], item['tags']])

    print(f"   ✅ {label}: 新增 {len(filtered_links)} 条链接")


def extract_community_id(url):
    match = re.search(r'/community/view/(\d+)', url)
    if match:
        return match.group(1)
    return None


# ================== 主程序 ==================
def main():
    print(f"🚀 启动 {CITY} 小区房源链接爬虫...")
    print(f"   ⏱️ 每个小区最多爬取 {MAX_LINKS_PER_PAGE} 条（二手房+租房各10条）")
    print(f"   ⏱️ 每个小区间隔 {SLEEP_RANGE[0]}-{SLEEP_RANGE[1]} 秒")

    if not os.path.exists(COMMUNITY_LINKS_CSV):
        print(f"❌ 未找到小区链接文件: {COMMUNITY_LINKS_CSV}")
        return

    # 1. 读取小区链接文件（包含小区URL和状态）
    df_links = pd.read_csv(COMMUNITY_LINKS_CSV, encoding='utf-8-sig')
    # 关键修复：将 house_状态 列中的 NaN 转为空字符串，确保筛选条件生效
    df_links['house_状态'] = df_links['house_状态'].fillna('').astype(str)

    print(f"📂 小区链接文件共有 {len(df_links)} 个小区")

    # 确保有 'house_状态' 列（已存在，但为保险）
    if 'house_状态' not in df_links.columns:
        df_links['house_状态'] = ''
        df_links.to_csv(COMMUNITY_LINKS_CSV, index=False, encoding='utf-8-sig')
        print("✅ 已添加 'house_状态' 列")

    # 2. 读取小区信息文件（包含小区名称）
    if not os.path.exists(COMMUNITY_INFO_CSV):
        print(f"⚠️ 未找到小区信息文件: {COMMUNITY_INFO_CSV}，将使用小区URL作为名称")
        community_name_map = {}
    else:
        df_info = pd.read_csv(COMMUNITY_INFO_CSV, encoding='utf-8-sig')
        print(f"📂 小区信息文件共有 {len(df_info)} 个小区")
        # 建立 小区URL -> 小区名称 的映射
        if 'community_url' in df_info.columns and 'community_name' in df_info.columns:
            community_name_map = dict(zip(df_info['community_url'], df_info['community_name']))
            print(f"📌 已建立 {len(community_name_map)} 个小区URL到名称的映射")
        else:
            print(f"⚠️ 小区信息文件中缺少 community_url 或 community_name 列，可用列: {df_info.columns.tolist()}")
            community_name_map = {}

    # 筛选待处理：状态为空或失败
    mask = (df_links['house_状态'] == '') | (df_links['house_状态'] == '失败')
    df_todo = df_links[mask].copy()
    print(f"📌 待处理小区: {len(df_todo)} 个")

    if df_todo.empty:
        print("✅ 所有小区已处理完成！")
        return

    driver = get_edge_driver()
    total_success = 0
    total_failed = 0

    for idx, row in df_todo.iterrows():
        community_url = row['小区URL']
        community_name = community_name_map.get(community_url, f"小区_{extract_community_id(community_url)}")
        community_id = extract_community_id(community_url)

        if not community_id:
            print(f"⚠️ 无法从URL提取小区ID: {community_url}，标记为失败")
            df_links.at[idx, 'house_状态'] = '失败'
            # 保存状态
            df_links.to_csv(COMMUNITY_LINKS_CSV, index=False, encoding='utf-8-sig')
            total_failed += 1
            continue

        print(f"\n[{total_success + total_failed + 1}/{len(df_todo)}] 处理小区: {community_name} (ID: {community_id})")

        # ---- 二手房 ----
        sale_url = f"https://{CITY_PINYIN}.anjuke.com/community/props/sale/{community_id}/"
        print(f"   🔍 爬取二手房第一页...")
        sale_links = extract_first_page_links(driver, sale_url, 'sale', community_id, community_name,
                                              MAX_LINKS_PER_PAGE)
        if sale_links:
            save_links_to_csv(sale_links, SALE_LINKS_CSV, '二手房', community_id, community_name)
        else:
            print(f"   📌 二手房: 无链接")

        # ---- 租房 ----
        rent_url = f"https://{CITY_PINYIN}.anjuke.com/community/props/rent/{community_id}/"
        print(f"   🔍 爬取租房第一页...")
        rent_links = extract_first_page_links(driver, rent_url, 'rent', community_id, community_name,
                                              MAX_LINKS_PER_PAGE)
        if rent_links:
            save_links_to_csv(rent_links, RENT_LINKS_CSV, '租房', community_id, community_name)
        else:
            print(f"   📌 租房: 无链接")

        # 更新状态
        df_links.at[idx, 'house_状态'] = '已爬取'
        df_links.to_csv(COMMUNITY_LINKS_CSV, index=False, encoding='utf-8-sig')
        total_success += 1
        print(f"   ✅ 小区处理完成")

        delay = SLEEP_RANGE[0] + random.random() * (SLEEP_RANGE[1] - SLEEP_RANGE[0])
        print(f"   ⏱️ 等待 {delay:.1f} 秒...")
        time.sleep(delay)

    driver.quit()
    print(f"\n✅ 爬取完成！")
    print(f"   成功: {total_success} 个小区，失败: {total_failed} 个")
    print(f"   📁 二手房链接: {SALE_LINKS_CSV}")
    print(f"   📁 租房链接: {RENT_LINKS_CSV}")


if __name__ == "__main__":
    main()