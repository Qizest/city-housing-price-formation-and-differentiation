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
from bs4 import BeautifulSoup

# ================== 配置 ==================
CITY = "金华"
CITY_PINYIN = "jinhua"
INPUT_CSV = os.path.join("..", "..", "data", CITY, "community", f"{CITY}小区_详情链接.csv")
OUTPUT_DIR = os.path.join("..", "..", "data", CITY, "community")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"{CITY}小区信息.csv")
STATUS_CSV = INPUT_CSV

MAX_RETRIES = 3
PAGE_LOAD_TIMEOUT = 45
SLEEP_RANGE = (2, 5)

EDGE_DRIVER_PATH = None


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


def restart_driver(driver):
    try:
        driver.quit()
    except:
        pass
    time.sleep(2)
    print("🔄 正在重启浏览器驱动...")
    return get_edge_driver()


# ================== 验证码检测 ==================
def check_for_captcha(driver):
    try:
        if driver.find_elements(By.CSS_SELECTOR, '.geetest_panel, .slide-verify, .captcha'):
            return True
        if any(k in driver.page_source for k in ['验证码', '滑动验证', '智能验证']):
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


def safe_fetch(driver, url, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            driver.get(url)
            time.sleep(3)
            if check_for_captcha(driver):
                if not handle_captcha(driver):
                    continue
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".info-list, .column-1"))
            )
            time.sleep(2)
            return True
        except Exception as e:
            print(f"⚠️ 加载失败 (尝试 {attempt + 1}/{retries}): {e}")
            time.sleep(3)
    return False


# ================== 提取周边配套（精简精准版） ==================
def extract_around_facilities(driver):
    """
    点击切换 tab，提取公交、地铁、学校、餐饮、购物、医院、银行
    返回: dict {category: list of (name, distance, detail)}
    """
    result = {}
    categories = ['bus', 'subway', 'school', 'restaurant', 'buy', 'hospital', 'bank']
    category_names = {
        'bus': '公交', 'subway': '地铁', 'school': '学校',
        'restaurant': '餐饮', 'buy': '购物', 'hospital': '医院', 'bank': '银行'
    }

    try:
        tab_box = driver.find_element(By.CSS_SELECTOR, '.tabBox .aroundTypeJs')
        tabs = tab_box.find_elements(By.CSS_SELECTOR, 'li.around')

        for tab in tabs:
            category = tab.get_attribute('data')
            if category not in categories:
                continue

            cat_name = category_names.get(category, category)
            print(f"      📌 提取 {cat_name}...")

            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", tab)
                time.sleep(0.3)
                driver.execute_script("arguments[0].click();", tab)

                # 等待数据出现（最多3秒，选择器精准）
                WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.itemTagBox .contanerBox .licontaner'))
                )
                time.sleep(0.5)  # 额外渲染缓冲

                items = driver.find_elements(By.CSS_SELECTOR, '.itemTagBox .contanerBox .licontaner')
                facilities = []
                for item in items:
                    try:
                        name_el = item.find_element(By.CSS_SELECTOR, '.itemTitle')
                        name = name_el.text.strip() if name_el else ''
                        dist_el = item.find_element(By.CSS_SELECTOR, '.itemdistance')
                        distance = dist_el.text.strip() if dist_el else ''
                        info_el = item.find_element(By.CSS_SELECTOR, '.itemInfo')
                        info = info_el.text.strip() if info_el else ''
                        if name:
                            facilities.append({'名称': name, '距离': distance, '详情': info})
                    except:
                        continue

                result[cat_name] = facilities
                print(f"         ✅ 提取到 {len(facilities)} 条")

            except Exception as e:
                print(f"         ⚠️ 提取 {cat_name} 失败: {e}")
                result[cat_name] = []

    except Exception as e:
        print(f"⚠️ 周边配套提取失败: {e}")

    return result


# ================== 解析小区详情页 ==================
def parse_community_page(soup, url, driver):
    data = {
        'community_id': re.search(r'/community/view/(\d+)', url).group(1) if re.search(r'/community/view/(\d+)',
                                                                                       url) else None,
        'community_url': url,
        'community_name': None,
        'location': None,
        'price': None,
        'build_year': None,
        'nsh': 0,
        'nrp': 0,
        'property_type': None,
        'ownership_type': None,
        'property_years': None,
        'total_households': None,
        'total_area': None,
        'plot_ratio': None,
        'greening_rate': None,
        'building_type': None,
        'business_circle': None,
        'heating': None,
        'supply_type': None,
        'parking': None,
        'property_fee': None,
        'parking_fee': None,
        'parking_management_fee': None,
        'property_company': None,
        'developer': None,
    }

    # 小区名称
    h1 = soup.find('h1')
    if h1:
        data['community_name'] = h1.get_text(strip=True)

    # 地址
    for div in soup.select('.column-1'):
        if div.find(string=re.compile('小区地址')):
            val = div.find('div', class_='value')
            if val:
                data['location'] = val.get_text(strip=True)
            break

    # 挂牌均价
    price = soup.find('span', class_='average')
    if price:
        m = re.search(r'(\d+\.?\d*)', price.get_text())
        if m:
            data['price'] = float(m.group(1))

    # 竣工时间
    for div in soup.select('.column-2'):
        if div.find(string=re.compile('竣工时间')):
            val = div.find('div', class_='value')
            if val:
                y = re.search(r'(\d{4})', val.get_text())
                if y:
                    data['build_year'] = int(y.group(1))
            break

    # 在售/在租数量
    sale = soup.find('div', class_='sale')
    if sale:
        num = sale.find('i', class_='source-number')
        if num:
            try:
                data['nsh'] = int(re.search(r'\d+', num.get_text()).group())
            except:
                pass

    rent = soup.find('div', class_='rent')
    if rent:
        num = rent.find('i', class_='source-number')
        if num:
            try:
                data['nrp'] = int(re.search(r'\d+', num.get_text()).group())
            except:
                pass

    # info-list 字段
    for item in soup.select('.column-1, .column-2'):
        label = item.find('div', class_='label')
        if label:
            ltext = label.get_text(strip=True)
            value = item.find('div', class_='value')
            if value:
                vtext = value.get_text(strip=True)
                if '物业类型' in ltext:
                    data['property_type'] = vtext
                elif '权属类别' in ltext:
                    data['ownership_type'] = vtext
                elif '产权年限' in ltext:
                    data['property_years'] = vtext
                elif '总户数' in ltext:
                    data['total_households'] = vtext
                elif any(x in ltext for x in ['总建面积', '建筑面积']):
                    data['total_area'] = vtext
                elif '容积率' in ltext:
                    data['plot_ratio'] = vtext
                elif '绿化率' in ltext:
                    data['greening_rate'] = vtext
                elif '建筑类型' in ltext:
                    data['building_type'] = vtext
                elif '所属商圈' in ltext:
                    data['business_circle'] = vtext
                elif '统一供暖' in ltext or '供暖' in ltext:
                    data['heating'] = vtext
                elif '供水供电' in ltext:
                    data['supply_type'] = vtext
                elif '停车位' in ltext:
                    data['parking'] = vtext
                elif '物业费' in ltext:
                    data['property_fee'] = vtext
                elif '停车费' in ltext:
                    data['parking_fee'] = vtext
                elif '车位管理费' in ltext:
                    data['parking_management_fee'] = vtext
                elif '物业公司' in ltext:
                    data['property_company'] = vtext
                elif '开发商' in ltext:
                    data['developer'] = vtext

    # 提取周边配套
    around = extract_around_facilities(driver)
    for cat, facilities in around.items():
        if facilities:
            data[f'around_{cat}'] = '; '.join([f"{f['名称']}({f['距离']})" for f in facilities[:10]])
            data[f'around_{cat}_count'] = len(facilities)
        else:
            data[f'around_{cat}'] = ''
            data[f'around_{cat}_count'] = 0

    return data


# ================== 主程序 ==================
def main():
    print(f"🚀 启动 {CITY} 小区详情爬虫...")

    if not os.path.exists(INPUT_CSV):
        print(f"❌ 未找到链接文件: {INPUT_CSV}")
        return

    df_links = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    if '状态' not in df_links.columns:
        df_links['状态'] = ''
    else:
        df_links['状态'] = df_links['状态'].fillna('').astype(str)

    mask = (df_links['状态'] == '') | (df_links['状态'] == '失败')
    df_todo = df_links[mask].copy()
    print(f"📄 共有 {len(df_todo)} 个小区待爬取")

    if len(df_todo) == 0:
        print("✅ 所有小区已处理完成！")
        return

    results = []
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV, encoding='utf-8-sig')
        results = existing_df.to_dict('records')
        print(f"📂 已有 {len(results)} 条小区记录")

    driver = get_edge_driver()
    total_success = 0
    total_failed = 0

    for idx, row in df_todo.iterrows():
        url = row['小区URL']
        tags = row.get('标签', '')
        print(f"\n[{total_success + total_failed + 1}/{len(df_todo)}] 爬取小区: {url}")

        if not safe_fetch(driver, url):
            total_failed += 1
            df_links.at[idx, '状态'] = '失败'
            df_links.to_csv(STATUS_CSV, index=False, encoding='utf-8-sig')
            continue

        try:
            soup = BeautifulSoup(driver.page_source, 'lxml')
            detail = parse_community_page(soup, url, driver)

            detail['tags'] = tags
            detail['爬取时间'] = time.strftime("%Y-%m-%d %H:%M:%S")

            results.append(detail)
            total_success += 1
            df_links.at[idx, '状态'] = '已爬取'
            df_links.to_csv(STATUS_CSV, index=False, encoding='utf-8-sig')

            print(f"   ✅ 成功: {detail.get('community_name', '未知')}")

            if len(results) % 5 == 0:
                # 中间保存也使用相同处理
                df_tmp = pd.DataFrame(results)
                if 'community_id' in df_tmp.columns:
                    df_tmp['community_id'] = df_tmp['community_id'].astype(str)
                df_tmp.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
                print(f"   💾 已保存 {len(results)} 条")

        except Exception as e:
            print(f"   ❌ 解析异常: {e}")
            total_failed += 1
            df_links.at[idx, '状态'] = '失败'
            df_links.to_csv(STATUS_CSV, index=False, encoding='utf-8-sig')

        time.sleep(SLEEP_RANGE[0] + random.random() * (SLEEP_RANGE[1] - SLEEP_RANGE[0]))

    driver.quit()

    # 最终保存，处理 community_id 防科学计数法
    df_result = pd.DataFrame(results)
    if 'community_id' in df_result.columns:
        df_result['community_id'] = df_result['community_id'].astype(str)
    df_result.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)

    print(f"\n✅ 爬取完成！共 {len(results)} 条小区信息")
    print(f"   保存至: {OUTPUT_CSV}")
    print(f"📊 本次成功: {total_success}，失败: {total_failed}")


if __name__ == "__main__":
    main()