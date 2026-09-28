# step5_floor.py
# 步骤5：楼层数据清洗提取（读取步骤4输出）
import re
import pandas as pd
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data_after_clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["杭州", "金华", "临沂"]
CATEGORIES = ["sale", "rent"]

VALID_FLOOR_TYPES = {'低层', '中层', '高层', '地下', '底层', '顶层', '平层'}


def parse_floor(floor_str):
    if not floor_str or pd.isna(floor_str):
        return ('', '')
    s = str(floor_str).strip()
    floor_type = ''
    total_floor = ''
    pattern = r'^(.*?)\(\s*(共\s*\d+\s*层)\s*\)$'
    match = re.match(pattern, s)
    if match:
        floor_type = match.group(1).strip()
        total_floor = match.group(2).strip()
        if not floor_type:
            floor_type = ''
        if floor_type and floor_type not in VALID_FLOOR_TYPES:
            floor_type = ''
        return (floor_type, total_floor)
    total_pattern = r'(共\s*\d+\s*层)'
    total_match = re.search(total_pattern, s)
    if total_match:
        total_floor = total_match.group(1).strip()
        rest = re.sub(total_pattern, '', s).strip()
        rest = re.sub(r'[()（）]', '', rest).strip()
        floor_type = rest if rest else ''
        if floor_type and floor_type not in VALID_FLOOR_TYPES:
            floor_type = ''
        return (floor_type, total_floor)
    floor_type = s
    if floor_type not in VALID_FLOOR_TYPES:
        floor_type = ''
    return (floor_type, '')


def process_city(city):
    print(f"\n--- 处理城市: {city} ---")
    for cat in CATEGORIES:
        house_type = "二手房" if cat == "sale" else "租房"
        # 优先读取步骤4的输出
        in_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_filled_property_ownership.csv"
        if not in_file.exists():
            in_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_filled_property.csv"
        if not in_file.exists():
            in_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_filled.csv"
        if not in_file.exists():
            print(f"⚠️ 房源文件不存在: {in_file}，跳过")
            continue

        df = pd.read_csv(in_file, encoding='utf-8-sig')

        if 'floor' in df.columns:
            parsed = df['floor'].apply(parse_floor)
            df['floor_type'] = parsed.apply(lambda x: x[0])
            df['total_floor'] = parsed.apply(lambda x: x[1])
            mask_invalid = (df['floor_type'] == '') & (df['total_floor'] == '')
            df.loc[mask_invalid, 'floor'] = ''
            print(f"   📊 无效楼层记录（已置空floor）: {mask_invalid.sum()} 条")
        else:
            df['floor_type'] = ''
            df['total_floor'] = ''
            print("   ⚠️ 原始数据无 'floor' 列")

        out_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_final_floor.csv"
        for col in ['house_id', 'community_id']:
            if col in df.columns:
                df[col] = df[col].astype(str)
        df.to_csv(out_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"   ✅ 楼层清洗完成，保存至: {out_file}")


def main():
    print("=" * 60)
    print("步骤5：楼层数据清洗提取（读取步骤4输出）")
    print("=" * 60)
    for city in CITIES:
        process_city(city)
    print("\n✅ 步骤5完成")


if __name__ == "__main__":
    main()