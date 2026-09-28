# step3_property.py
# 步骤3：物业类型标准化（先标准化房源，再以小区物业类型覆盖）
import pandas as pd
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data_after_clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["杭州", "金华", "临沂"]
CATEGORIES = ["sale", "rent"]


def standardize_property_type(val):
    if not val or pd.isna(val):
        return '其他'
    s = str(val).strip()
    if s in ['普通住宅', '公寓', '别墅', '排屋', '平房', '其他']:
        return s
    if '公寓' in s:
        return '公寓'
    if '别墅' in s:
        return '别墅'
    if '排屋' in s:
        return '排屋'
    if '平房' in s:
        return '平房'
    if '住宅' in s:
        return '普通住宅'
    return '其他'


def unify_property_type_with_community(city):
    print(f"\n--- 处理城市: {city} ---")

    comm_file = OUTPUT_DIR / city / "community" / f"{city}小区信息_cleaned.csv"
    if not comm_file.exists():
        print(f"⚠️ 小区清洗文件不存在: {comm_file}，跳过")
        return
    df_comm = pd.read_csv(comm_file, encoding='utf-8-sig')
    if 'community_id' not in df_comm.columns or 'property_type' not in df_comm.columns:
        print("⚠️ 小区数据缺少 community_id 或 property_type，跳过")
        return
    comm_prop = df_comm[['community_id', 'property_type']].dropna(subset=['property_type'])
    comm_prop = comm_prop.drop_duplicates(subset=['community_id'], keep='first')
    prop_map = dict(zip(comm_prop['community_id'].astype(str).str.strip(),
                        comm_prop['property_type']))
    print(f"   📌 从小区加载物业类型映射 {len(prop_map)} 条")

    for cat in CATEGORIES:
        house_type = "二手房" if cat == "sale" else "租房"
        in_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_filled.csv"
        if not in_file.exists():
            print(f"⚠️ 房源填充文件不存在: {in_file}，跳过")
            continue
        df_house = pd.read_csv(in_file, encoding='utf-8-sig')

        if 'property_type' in df_house.columns:
            df_house['property_type'] = df_house['property_type'].apply(standardize_property_type)
        else:
            df_house['property_type'] = '其他'
            print("   ⚠️ 房源数据缺少 property_type 列，已填充为 '其他'")

        if 'community_id' in df_house.columns:
            df_house['community_id'] = df_house['community_id'].astype(str).str.strip()
            df_house['property_type_from_comm'] = df_house['community_id'].map(prop_map)
            mask = df_house['property_type_from_comm'].notna() & (df_house['property_type_from_comm'] != '')
            df_house.loc[mask, 'property_type'] = df_house.loc[mask, 'property_type_from_comm']
            df_house = df_house.drop(columns=['property_type_from_comm'])
        else:
            print("   ⚠️ 房源数据缺少 community_id，无法进行小区覆盖")

        out_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_filled_property.csv"
        for col in ['house_id', 'community_id']:
            if col in df_house.columns:
                df_house[col] = df_house[col].astype(str)
        df_house.to_csv(out_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"   ✅ 物业类型标准化并覆盖完成，保存至: {out_file}")


def main():
    print("=" * 60)
    print("步骤3：物业类型标准化与小区覆盖")
    print("=" * 60)
    for city in CITIES:
        unify_property_type_with_community(city)
    print("\n✅ 步骤3完成")


if __name__ == "__main__":
    main()