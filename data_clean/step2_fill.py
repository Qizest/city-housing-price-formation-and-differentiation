# step2_fill.py
# 步骤2：缺失值填充（district、build_year、电梯两轮填充）
import os
import re
import csv
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data_after_clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["杭州", "金华", "临沂"]
CATEGORIES = ["sale", "rent"]

HAS_ELEVATOR_KEYWORDS = [
    '电梯房', '有电梯', '带电梯', '电梯直达', '电梯入户',
    '电梯花园', '电梯洋房', '电梯公寓', '电梯高层', '电梯小高'
]
NO_ELEVATOR_KEYWORDS = [
    '无电梯', '没电梯', '没有电梯', '不带电梯',
    '步梯房', '楼梯房', '爬楼梯', '电梯加装', '电梯待装',
    '电梯规划', '电梯筹备', '电梯申请', '电梯集资', '电梯改造'
]


def extract_community_id_from_url(url):
    if not url or pd.isna(url):
        return None
    match = re.search(r'/community/view/(\d+)', str(url))
    return match.group(1) if match else None


def extract_elevator_from_text(text):
    if not text or pd.isna(text):
        return None
    text = str(text).lower()
    has_any = any(kw in text for kw in HAS_ELEVATOR_KEYWORDS)
    no_any = any(kw in text for kw in NO_ELEVATOR_KEYWORDS)
    if has_any and no_any:
        return None
    for kw in NO_ELEVATOR_KEYWORDS:
        if kw in text:
            return '无电梯'
    for kw in HAS_ELEVATOR_KEYWORDS:
        if kw in text:
            return '有电梯'
    return None


def load_district_and_year_maps(city):
    links_file = DATA_DIR / city / "community" / f"{city}小区_详情链接.csv"
    district_map = {}
    if links_file.exists():
        df_links = pd.read_csv(links_file, encoding='utf-8-sig')
        if '小区URL' in df_links.columns and 'district' in df_links.columns:
            for _, row in df_links.iterrows():
                url = str(row['小区URL']).strip()
                district = str(row['district']).strip() if pd.notna(row['district']) else ''
                if url and district:
                    comm_id = extract_community_id_from_url(url)
                    if comm_id:
                        district_map[comm_id] = district
        print(f"   📌 从小区链接表加载 district 映射: {len(district_map)} 条")

    info_file = DATA_DIR / city / "community" / f"{city}小区信息.csv"
    year_map = {}
    if info_file.exists():
        df_info = pd.read_csv(info_file, encoding='utf-8-sig')
        if 'community_id' in df_info.columns and 'build_year' in df_info.columns:
            for _, row in df_info.iterrows():
                comm_id = str(row['community_id']).strip() if pd.notna(row['community_id']) else ''
                year = row['build_year']
                if comm_id and pd.notna(year) and year != '':
                    year_map[comm_id] = year
        print(f"   📌 从小区信息表加载 build_year 映射: {len(year_map)} 条")
    else:
        print(f"   ⚠️ 小区信息文件不存在，无法填充 build_year")
    return district_map, year_map


def fill_elevator_by_community_consistency(df_house):
    if 'community_id' not in df_house.columns or 'elevator' not in df_house.columns:
        print("⚠️ 缺少 community_id 或 elevator 列，跳过第二轮填充")
        return df_house
    df_house['community_id'] = df_house['community_id'].astype(str).str.strip()
    df_house['elevator'] = df_house['elevator'].replace(['', 'None', 'nan'], None)
    df_house['elevator'] = df_house['elevator'].where(pd.notna(df_house['elevator']), None)
    non_null = df_house[df_house['elevator'].notna()]
    if non_null.empty:
        print("   ℹ️ 所有电梯字段均为空，无法进行一致性填充")
        return df_house
    comm_elevator = non_null.groupby('community_id')['elevator'].unique()
    consistent_comms = comm_elevator[comm_elevator.apply(len) == 1].index.tolist()
    print(f"   ℹ️ 一致小区数量: {len(consistent_comms)}")
    consistent_map = {}
    for comm in consistent_comms:
        val = non_null[non_null['community_id'] == comm]['elevator'].iloc[0]
        consistent_map[comm] = val
    sample = dict(list(consistent_map.items())[:5])
    print(f"   ℹ️ 一致小区示例: {sample}")
    mask = df_house['elevator'].isna()
    if mask.any():
        df_house.loc[mask, 'elevator'] = df_house.loc[mask, 'community_id'].map(consistent_map)
        filled_count = mask.sum() - df_house.loc[mask, 'elevator'].isna().sum()
        print(f"   ✅ 基于小区一致性填充电梯 {filled_count} 条")
    else:
        print("   ℹ️ 没有需要填充的空值")
    return df_house


def fill_city(city):
    print(f"\n{'='*60}")
    print(f"填充城市: {city}")
    print(f"{'='*60}")

    district_map, year_map = load_district_and_year_maps(city)
    print(f"📌 district映射大小: {len(district_map)}，build_year映射大小: {len(year_map)}")

    all_dfs = []
    types = []
    for cat in CATEGORIES:
        house_type = "二手房" if cat == "sale" else "租房"
        cleaned_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_cleaned.csv"
        if cleaned_file.exists():
            df = pd.read_csv(cleaned_file, encoding='utf-8-sig')
            df['_type'] = cat
            all_dfs.append(df)
            types.append(cat)
            print(f"📂 加载 {house_type} 清洗数据: {len(df)} 条")
        else:
            print(f"⚠️ 文件不存在: {cleaned_file}")

    if not all_dfs:
        print("❌ 没有可填充的数据")
        return

    df_merged = pd.concat(all_dfs, ignore_index=True)
    print(f"📊 合并后总记录: {len(df_merged)}")

    if 'community_id' not in df_merged.columns:
        if 'community_url' in df_merged.columns:
            df_merged['community_id'] = df_merged['community_url'].apply(extract_community_id_from_url)
        else:
            print("⚠️ 数据缺少 community_id 和 community_url，无法进行关联填充")
            return
    df_merged['community_id'] = df_merged['community_id'].astype(str).str.strip()

    # 2.1 覆盖 district
    if district_map and 'community_id' in df_merged.columns:
        if 'district' not in df_merged.columns:
            df_merged['district'] = ''
        df_merged['district'] = df_merged['community_id'].map(district_map).fillna('')
        filled_district = df_merged['community_id'].map(district_map).notna().sum()
        print(f"   ✅ 覆盖 district {filled_district} 条（未匹配的已置空）")
    else:
        print("⚠️ 无法覆盖 district：缺少映射或 community_id")

    # 2.2 填充 build_year
    if year_map and 'community_id' in df_merged.columns:
        mask = df_merged['build_year'].isna() | (df_merged['build_year'] == '') | (df_merged['build_year'] == 'None')
        if mask.any():
            filled = df_merged.loc[mask, 'community_id'].map(year_map)
            df_merged.loc[mask, 'build_year'] = filled
            print(f"   ✅ 填充 build_year 缺失 {mask.sum()} 条")
    else:
        print("⚠️ 无法填充 build_year：缺少映射或 community_id")

    # 2.3 第一轮电梯填充
    text_fields = ['house_title', 'tags', 'page_tags']
    for f in text_fields:
        if f not in df_merged.columns:
            df_merged[f] = ''

    elevator_col = 'elevator' if 'elevator' in df_merged.columns else None

    def fill_elevator_row(row):
        if elevator_col and row.get(elevator_col) and pd.notna(row[elevator_col]) and str(row[elevator_col]).strip() not in ['', 'None', 'nan']:
            return row[elevator_col]
        combined = ' '.join([
            str(row.get('house_title', '')),
            str(row.get('tags', '')),
            str(row.get('page_tags', ''))
        ])
        return extract_elevator_from_text(combined)

    df_merged['elevator_filled'] = df_merged.apply(fill_elevator_row, axis=1)

    if elevator_col:
        mask_empty = df_merged[elevator_col].isna() | (df_merged[elevator_col] == '') | (df_merged[elevator_col] == 'None')
        df_merged.loc[mask_empty, elevator_col] = df_merged.loc[mask_empty, 'elevator_filled']
        df_merged = df_merged.drop(columns=['elevator_filled'])
    else:
        df_merged['elevator'] = df_merged['elevator_filled']
        df_merged = df_merged.drop(columns=['elevator_filled'])

    print(f"   ✅ 第一轮电梯填充完成")

    df_merged = fill_elevator_by_community_consistency(df_merged)

    for cat in types:
        house_type = "二手房" if cat == "sale" else "租房"
        df_type = df_merged[df_merged['_type'] == cat].drop(columns=['_type'])
        output_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_filled.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        for col in ['house_id', 'community_id']:
            if col in df_type.columns:
                df_type[col] = df_type[col].astype(str)
        df_type.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"✅ 保存 {house_type} 填充后数据: {output_file}")


def main():
    print("=" * 60)
    print("步骤2：缺失值填充")
    print("=" * 60)
    for city in CITIES:
        fill_city(city)
    print("\n✅ 步骤2完成")


if __name__ == "__main__":
    main()