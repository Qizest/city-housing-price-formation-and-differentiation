# step1_clean.py
# 步骤1：去重与文本清洗
import os
import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data_after_clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["杭州", "金华", "临沂"]
CATEGORIES = ["sale", "rent"]


def extract_house_id_from_url(url, category):
    if not url or pd.isna(url):
        return None
    url = str(url).strip()
    if category == 'sale':
        match = re.search(r'/prop/view/([A-Za-z0-9]+)', url)
        if not match:
            match = re.search(r'/sale/(\d+)', url)
        return match.group(1) if match else None
    else:
        match = re.search(r'/fangyuan/(\d+)', url)
        return match.group(1) if match else None


def clean_dataframe(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace('\n', ' ').str.replace('\r', ' ')
    return df


def deduplicate(df, id_col):
    if id_col not in df.columns:
        return df

    def count_non_empty(row):
        return row.notna().sum() + (row.astype(str) != '').sum()

    df['_non_empty_count'] = df.apply(count_non_empty, axis=1)
    df_sorted = df.sort_values('_non_empty_count', ascending=False)
    df_dedup = df_sorted.drop_duplicates(subset=[id_col], keep='first')
    df_dedup = df_dedup.drop(columns=['_non_empty_count'])
    return df_dedup


def clean_house_data(city, category):
    house_type = "二手房" if category == "sale" else "租房"
    raw_file = DATA_DIR / city / category / f"{city}{house_type}_房源信息.csv"
    output_file = OUTPUT_DIR / city / category / f"{city}{house_type}_房源信息_cleaned.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not raw_file.exists():
        print(f"⚠️ 原始房源文件不存在: {raw_file}")
        return None

    df = pd.read_csv(raw_file, encoding='utf-8-sig')
    print(f"📂 读取房源 {len(df)} 条")

    df = clean_dataframe(df)
    if 'house_id' not in df.columns:
        if '详情页URL' in df.columns:
            df['house_id'] = df['详情页URL'].apply(lambda x: extract_house_id_from_url(x, category))
        else:
            print("❌ 缺少 house_id 和 详情页URL，无法去重")
            return None

    df = deduplicate(df, 'house_id')
    print(f"  去重后 {len(df)} 条")

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 清洗后房源数据保存至: {output_file}")
    return df


def clean_community_data(city):
    raw_file = DATA_DIR / city / "community" / f"{city}小区信息.csv"
    output_file = OUTPUT_DIR / city / "community" / f"{city}小区信息_cleaned.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not raw_file.exists():
        print(f"⚠️ 原始小区文件不存在: {raw_file}")
        return None

    df = pd.read_csv(raw_file, encoding='utf-8-sig')
    print(f"📂 读取小区 {len(df)} 条")

    df = clean_dataframe(df)
    id_col = 'community_id' if 'community_id' in df.columns else ('小区ID' if '小区ID' in df.columns else None)
    if id_col:
        df = deduplicate(df, id_col)
        print(f"  去重后 {len(df)} 条")
    else:
        print("⚠️ 缺少 community_id，跳过小区去重")

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 清洗后小区数据保存至: {output_file}")
    return df


def main():
    print("=" * 60)
    print("步骤1：去重与文本清洗")
    print("=" * 60)
    for city in CITIES:
        print(f"\n--- 清洗城市: {city} ---")
        for cat in CATEGORIES:
            clean_house_data(city, cat)
        clean_community_data(city)
    print("\n✅ 步骤1完成")


if __name__ == "__main__":
    main()