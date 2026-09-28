# step7_extract.py
# 步骤7：完整数据提取（房源严检 + 小区严检）
# 修改：先过滤有效小区，再以此过滤房源，确保数据一致性
import pandas as pd
import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data_after_clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["杭州", "金华", "临沂"]
CATEGORIES = ["sale", "rent"]

# ================== 字段定义（含产权字段） ==================
REQUIRED_HOUSE_COLS = [
    'house_id', 'house_title', 'city', 'district', 'community_name', 'house_type',
    'layout', 'area', 'total_price', 'floor', 'orientation', 'decoration',
    'build_year', 'elevator', 'property_type', 'ownership_type', 'property_years'
]

REQUIRED_COMMUNITY_COLS = [
    'community_name', 'tag', 'location', 'price', 'build_year', 'nsh', 'nrp',
    'ownership_type', 'property_years'
]


# ================== 有效性检查 ==================
def is_valid_house_row(row):
    for col in REQUIRED_HOUSE_COLS:
        val = row.get(col)
        if pd.isna(val):
            return False
        s = str(val).strip()
        if s == '' or s.lower() in ('nan', 'none'):
            return False
    # 楼层检查
    floor_type = str(row.get('floor_type', '')).strip()
    total_floor = str(row.get('total_floor', '')).strip()
    if floor_type == '' and total_floor == '':
        return False
    return True


def is_valid_community_row(row):
    for col in REQUIRED_COMMUNITY_COLS:
        val = row.get(col)
        if pd.isna(val):
            return False
        s = str(val).strip()
        if s == '' or s.lower() in ('nan', 'none'):
            return False
    return True


# ================== 辅助函数 ==================
def aggregate_community_tags(all_house_dfs):
    """从所有房源数据中聚合每个小区的 tags 和 page_tags"""
    tag_map = defaultdict(set)
    for df in all_house_dfs:
        if 'community_name' not in df.columns:
            continue
        df_clean = df[['community_name', 'tags', 'page_tags']].copy()
        df_clean = df_clean.dropna(subset=['community_name'])
        for _, row in df_clean.iterrows():
            comm = str(row['community_name']).strip()
            if not comm:
                continue
            tags = []
            if pd.notna(row.get('tags')):
                tags.extend(str(row['tags']).split('|'))
            if pd.notna(row.get('page_tags')):
                tags.extend(str(row['page_tags']).split('|'))
            clean_tags = [t.strip() for t in tags if t.strip()]
            tag_map[comm].update(clean_tags)
    result = {}
    for comm, tags_set in tag_map.items():
        result[comm] = '|'.join(sorted(tags_set)) if tags_set else ''
    return result


def save_dataframe_with_quoted_ids(df, output_path, id_cols=['house_id', 'community_id']):
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df.to_csv(output_path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)


# ================== 主程序 ==================
def main():
    print("=" * 60)
    print("步骤7：完整数据提取（含产权字段）")
    print("=" * 60)

    HOUSE_OUTPUT = OUTPUT_DIR / "完整房源数据.csv"
    COMMUNITY_OUTPUT = OUTPUT_DIR / "完整小区数据.csv"

    all_house_dfs = []
    all_comm_dfs = []

    for city in CITIES:
        print(f"\n处理城市: {city}")

        # ---------- 加载小区数据（优先使用步骤4清洗后的版本） ----------
        comm_file = OUTPUT_DIR / city / "community" / f"{city}小区信息_ownership_cleaned.csv"
        if not comm_file.exists():
            comm_file = OUTPUT_DIR / city / "community" / f"{city}小区信息_cleaned.csv"
        if not comm_file.exists():
            print(f"⚠️ 小区文件不存在: {comm_file}")
            continue

        df_comm = pd.read_csv(comm_file, encoding='utf-8-sig')
        # 自动识别小区名称列
        if 'community_name' not in df_comm.columns:
            for col in ['小区名称', '小区名', 'name', 'community']:
                if col in df_comm.columns:
                    df_comm.rename(columns={col: 'community_name'}, inplace=True)
                    break
            if 'community_name' not in df_comm.columns:
                print(f"⚠️ 小区表 {comm_file} 中无法识别小区名称列，跳过")
                continue

        # 确保产权字段存在，若缺失则补默认值
        if 'ownership_type' not in df_comm.columns:
            df_comm['ownership_type'] = '未知'
        if 'property_years' not in df_comm.columns:
            df_comm['property_years'] = 70

        df_comm['city'] = city
        all_comm_dfs.append(df_comm)

        # ---------- 加载房源数据（优先使用步骤6的输出） ----------
        for cat in CATEGORIES:
            house_type = "二手房" if cat == "sale" else "租房"
            # 文件读取优先级
            final_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_final_floor_ori.csv"
            if not final_file.exists():
                final_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_final_floor.csv"
            if not final_file.exists():
                final_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_filled_property_ownership.csv"
            if not final_file.exists():
                final_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_filled_property.csv"
            if not final_file.exists():
                final_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_filled.csv"
            if not final_file.exists():
                print(f"⚠️ 房源文件不存在: {final_file}")
                continue

            df_house = pd.read_csv(final_file, encoding='utf-8-sig')

            # 补全可能缺失的字段
            if 'floor_type' not in df_house.columns:
                df_house['floor_type'] = ''
            if 'total_floor' not in df_house.columns:
                df_house['total_floor'] = ''
            if 'ownership_type' not in df_house.columns:
                df_house['ownership_type'] = ''
            if 'property_years' not in df_house.columns:
                df_house['property_years'] = ''

            # 确保 community_name 存在
            if 'community_name' not in df_house.columns:
                for col in ['小区名称', '小区名', 'community']:
                    if col in df_house.columns:
                        df_house.rename(columns={col: 'community_name'}, inplace=True)
                        break
                if 'community_name' not in df_house.columns:
                    df_house['community_name'] = ''

            df_house['city'] = city
            df_house['house_type'] = house_type
            all_house_dfs.append(df_house)

    if not all_house_dfs and not all_comm_dfs:
        print("❌ 没有找到任何数据")
        return

    # ---------- 聚合小区 tag ----------
    tag_map = aggregate_community_tags(all_house_dfs)

    # ========== 【修改】先处理小区数据，获得有效小区ID集合 ==========
    print("\n📊 处理小区数据...")
    comm_records = []
    valid_community_ids = set()   # 存放通过有效性检查的小区ID

    for df in all_comm_dfs:
        df['tag'] = df['community_name'].map(tag_map).fillna('')
        for col in REQUIRED_COMMUNITY_COLS:
            if col not in df.columns:
                df[col] = ''
        mask = df.apply(is_valid_community_row, axis=1)
        valid = df[mask].copy()
        comm_records.append(valid)
        print(f"  {valid.shape[0]}/{df.shape[0]} 条有效小区")

    if comm_records:
        final_comm = pd.concat(comm_records, ignore_index=True)
        final_comm = final_comm.drop_duplicates(subset=['community_name'], keep='first')
        # 收集有效的小区ID（优先使用 community_id，若不存在则用名称）
        if 'community_id' in final_comm.columns:
            valid_community_ids = set(final_comm['community_id'].astype(str).str.strip())
        else:
            # 若没有ID，则使用小区名称作为关联键（但建议有ID）
            valid_community_names = set(final_comm['community_name'].astype(str).str.strip())
        # 保存小区表
        if 'community_id' in final_comm.columns:
            final_comm['community_id'] = final_comm['community_id'].astype(str)
        save_dataframe_with_quoted_ids(final_comm, COMMUNITY_OUTPUT, ['community_id'])
        print(f"\n✅ 完整小区数据已保存: {COMMUNITY_OUTPUT} ({len(final_comm)} 条)")

    # ========== 【修改】处理房源数据，只保留有效小区 ==========
    print("\n📊 处理房源数据...")
    house_records = []
    for df in all_house_dfs:
        # 补全列
        for col in REQUIRED_HOUSE_COLS:
            if col not in df.columns:
                df[col] = ''

        # 【新增】根据有效小区ID过滤
        if 'community_id' in df.columns:
            df['community_id'] = df['community_id'].astype(str).str.strip()
            if valid_community_ids:   # 如果有有效ID集合，则过滤
                df = df[df['community_id'].isin(valid_community_ids)]
            # 若没有ID集合（理论上不会），可用名称过滤
            # elif valid_community_names:
            #     df['community_name'] = df['community_name'].astype(str).str.strip()
            #     df = df[df['community_name'].isin(valid_community_names)]

        # 然后进行原有的有效性检查
        mask = df.apply(is_valid_house_row, axis=1)
        valid = df[mask].copy()
        house_records.append(valid)
        print(f"  {valid.shape[0]}/{df.shape[0]} 条有效房源（过滤后）")

    if house_records:
        final_house = pd.concat(house_records, ignore_index=True)
        final_house = final_house.drop_duplicates(subset=['house_id'], keep='first')
        save_dataframe_with_quoted_ids(final_house, HOUSE_OUTPUT, ['house_id'])
        print(f"\n✅ 完整房源数据已保存: {HOUSE_OUTPUT} ({len(final_house)} 条)")

    print("\n✅ 步骤7完成")


if __name__ == "__main__":
    main()