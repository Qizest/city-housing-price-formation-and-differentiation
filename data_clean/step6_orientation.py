# step6_orientation.py
# 步骤6：朝向清洗（统一为东、南、西、北）
import pandas as pd
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data_after_clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["杭州", "金华", "临沂"]
CATEGORIES = ["sale", "rent"]


def clean_orientation(ori_str):
    if not ori_str or pd.isna(ori_str):
        return ''
    s = str(ori_str).strip()
    has_north = '北' in s
    has_south = '南' in s
    has_east = '东' in s
    has_west = '西' in s
    dirs = []
    if has_north:
        dirs.append('北')
    if has_south:
        dirs.append('南')
    if has_east:
        dirs.append('东')
    if has_west:
        dirs.append('西')
    if len(dirs) == 1:
        return dirs[0]
    return s


def process_city(city):
    print(f"\n--- 处理城市: {city} ---")
    for cat in CATEGORIES:
        house_type = "二手房" if cat == "sale" else "租房"
        in_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_final_floor.csv"
        if not in_file.exists():
            print(f"⚠️ 文件不存在: {in_file}，跳过")
            continue
        df = pd.read_csv(in_file, encoding='utf-8-sig')
        if 'orientation' in df.columns:
            df['orientation'] = df['orientation'].apply(clean_orientation)
        else:
            print(f"   ⚠️ 房源表无 'orientation' 列，跳过清洗")
        out_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_final_floor_ori.csv"
        for col in ['house_id', 'community_id']:
            if col in df.columns:
                df[col] = df[col].astype(str)
        df.to_csv(out_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"   ✅ 朝向清洗完成，保存至: {out_file}")


def main():
    print("=" * 60)
    print("步骤6：朝向清洗（统一为东、南、西、北）")
    print("=" * 60)
    for city in CITIES:
        process_city(city)
    print("\n✅ 步骤6完成")


if __name__ == "__main__":
    main()