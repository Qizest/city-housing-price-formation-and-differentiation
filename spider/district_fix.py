import time
import os
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
CITIES = ["杭州", "金华", "临沂"]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
DATA_DIR = os.path.join(BASE_DIR, "data")

# 每个城市的行政区关键词（仅用于校验，非必须）
CITY_DISTRICTS = {
    "杭州": ["滨江", "西湖", "拱墅", "上城", "萧山", "余杭", "临平", "钱塘", "富阳", "临安", "淳安", "建德", "桐庐"],
    "金华": ["婺城", "金东", "东阳", "永康", "兰溪", "武义", "浦江", "磐安"],
    "临沂": ["兰山", "北城新区", "罗庄", "河东", "开发区", "临沭", "郯城", "沂水", "兰陵", "莒南", "平邑", "费县", "沂南", "蒙阴", "高新区"]
}

# 驱动配置
EDGE_DRIVER_PATH = None
PAGE_LOAD_TIMEOUT = 45
SLEEP_RANGE = (3, 6)


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
        if '安全验证' in driver.page_source or (driver.title and '安全验证' in driver.title):
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


def safe_fetch(driver, url, retries=3):
    for attempt in range(retries):
        try:
            driver.get(url)
            time.sleep(2)
            if check_for_captcha(driver):
                if not handle_captcha(driver):
                    continue
            # 等待标题出现
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'h1.title'))
            )
            return True
        except Exception as e:
            print(f"⚠️ 加载失败 (尝试 {attempt+1}/{retries}): {e}")
            time.sleep(3)
    return False


# ================== 提取行政区 ==================
def extract_district_from_page(driver):
    """
    从小区详情页提取行政区（如"上城"）
    返回 district 或 None
    """
    try:
        # 查找 .sub-title
        sub_title = driver.find_element(By.CSS_SELECTOR, 'p.sub-title')
        text = sub_title.text.strip()
        # 格式通常是 "上城-南星-钱江路158号"
        # 取第一个 '-' 之前的部分
        if '-' in text:
            district = text.split('-')[0].strip()
            return district
        else:
            # 没有分隔符，尝试直接用整个文本
            return text
    except:
        return None


# ================== 处理单个城市 ==================
def process_city(city):
    print(f"\n{'='*60}")
    print(f"处理 {city} 小区行政区补爬")
    print(f"{'='*60}")

    community_csv = os.path.join(DATA_DIR, city, "community", f"{city}小区_详情链接.csv")
    if not os.path.exists(community_csv):
        print(f"⚠️ 文件不存在: {community_csv}")
        return

    df = pd.read_csv(community_csv, encoding='utf-8-sig')
    # 确保有 '小区URL' 列
    if '小区URL' not in df.columns:
        print("⚠️ 缺少 '小区URL' 列，跳过")
        return

    # 确保有 '状态' 列（用于断点续爬）
    if '状态' not in df.columns:
        df['状态'] = ''
    # 确保有 'district' 列
    if 'district' not in df.columns:
        df['district'] = ''

    # 筛选待处理：状态为空或失败，或者 district 为空
    mask = (df['状态'] == '') | (df['状态'] == '失败') | (df['district'].isna()) | (df['district'] == '')
    df_todo = df[mask].copy()
    print(f"📌 待处理小区数: {len(df_todo)}")

    if df_todo.empty:
        print("✅ 所有小区行政区已补爬完成")
        return

    driver = get_edge_driver()
    total_success = 0
    total_failed = 0

    for idx, row in df_todo.iterrows():
        url = row['小区URL']
        print(f"\n[{total_success + total_failed + 1}/{len(df_todo)}] 处理: {url}")

        if not safe_fetch(driver, url):
            total_failed += 1
            df.at[idx, '状态'] = '失败'
            df.to_csv(community_csv, index=False, encoding='utf-8-sig')
            continue

        district = extract_district_from_page(driver)
        if district:
            df.at[idx, 'district'] = district
            df.at[idx, '状态'] = '已爬取'
            total_success += 1
            print(f"   ✅ 提取到行政区: {district}")
        else:
            total_failed += 1
            df.at[idx, '状态'] = '失败'
            print(f"   ❌ 无法提取行政区")

        # 每处理5个保存一次
        if (total_success + total_failed) % 5 == 0:
            df.to_csv(community_csv, index=False, encoding='utf-8-sig')
            print(f"   💾 已保存中间状态")

        time.sleep(SLEEP_RANGE[0] + random.random() * (SLEEP_RANGE[1] - SLEEP_RANGE[0]))

    driver.quit()

    # 最终保存
    df.to_csv(community_csv, index=False, encoding='utf-8-sig')
    print(f"✅ {city} 处理完成！成功: {total_success}，失败: {total_failed}")


# ================== 主程序 ==================
def main():
    print("🚀 启动行政区补爬脚本...")
    for city in CITIES:
        process_city(city)
    print("\n🎉 所有城市行政区补爬完成！")

if __name__ == "__main__":
    main()