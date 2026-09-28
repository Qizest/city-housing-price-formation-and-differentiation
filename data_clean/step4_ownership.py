# step4_ownership.py
# 步骤4：产权类型与产权年限清洗
# 直接覆盖原列 ownership_type 和 property_years（不创建任何新列）
import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data_after_clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["杭州", "金华", "临沂"]
CATEGORIES = ["sale", "rent"]


def clean_ownership_type(val):
    """产权类型归并为5大类，直接覆盖原值"""
    if not val or pd.isna(val):
        return '未知'
    s = str(val).strip()
    if '商品房住宅' in s:
        return '纯商品住宅'
    if '动迁配套房' in s or '动迁安置房' in s:
        return '保障/政策类住宅'
    if '经济适用房' in s:
        return '保障/政策类住宅'
    if '单位集体自建房' in s:
        return '保障/政策类住宅'
    if '商业' in s or '办公' in s:
        return '商办类'
    if '公租房和廉租房' in s:
        return '公共租赁房'
    if '小产权房' in s:
        return '高风险/无产权'
    if '集体租赁住房和城中村' in s:
        return '高风险/无产权'
    return '其他'


def extract_property_years(val):
    """从产权年限字符串中提取数值，直接覆盖原值"""
    if not val or pd.isna(val):
        return None
    s = str(val).strip()
    match = re.search(r'(\d+)', s)
    if match:
        return int(match.group(1))
    return None


def get_mode(series):
    """计算众数"""
    if series.empty or series.isna().all():
        return None
    mode_vals = series.mode()
    if len(mode_vals) > 0:
        return mode_vals[0]
    return None


def process_city(city):
    print(f"\n--- 处理城市: {city} ---")

    # ============================================================
    # 第一部分：清洗小区表（直接在原列上操作）
    # ============================================================
    comm_file = OUTPUT_DIR / city / "community" / f"{city}小区信息_cleaned.csv"
    if not comm_file.exists():
        print(f"⚠️ 小区信息文件不存在: {comm_file}，跳过")
        return

    df_comm = pd.read_csv(comm_file, encoding='utf-8-sig')

    if 'community_id' not in df_comm.columns:
        print("⚠️ 小区表缺少 community_id，跳过")
        return

    print(f"   📊 小区表记录数: {len(df_comm)}")

    # ---- 直接覆盖 ownership_type 列 ----
    if 'ownership_type' in df_comm.columns:
        df_comm['ownership_type'] = df_comm['ownership_type'].apply(clean_ownership_type)
        print(f"\n   📌 产权类型分布（清洗后）:")
        print(df_comm['ownership_type'].value_counts().to_string())
    else:
        print("⚠️ 小区表缺少 ownership_type 列，将创建该列并设为'未知'")
        df_comm['ownership_type'] = '未知'

    # ---- 直接覆盖 property_years 列（提取数值） ----
    if 'property_years' in df_comm.columns:
        df_comm['property_years'] = df_comm['property_years'].apply(extract_property_years)
    else:
        print("⚠️ 小区表缺少 property_years 列，将创建该列并设为 None")
        df_comm['property_years'] = None

    # ---- 按产权类型计算众数（用于填充缺失值） ----
    type_mode_years = {}
    for otype in df_comm['ownership_type'].unique():
        if pd.isna(otype):
            continue
        sub = df_comm[df_comm['ownership_type'] == otype]['property_years']
        mode_val = get_mode(sub)
        if mode_val is not None:
            type_mode_years[otype] = mode_val

    global_mode = get_mode(df_comm['property_years'])
    if global_mode is None:
        global_mode = 70

    for otype in df_comm['ownership_type'].unique():
        if pd.isna(otype):
            continue
        if otype not in type_mode_years:
            type_mode_years[otype] = global_mode

    print(f"\n   📌 各产权类型对应的产权年限众数:")
    for k, v in type_mode_years.items():
        print(f"      {k}: {v}年")

    # ---- 填充 property_years 缺失值（按产权类型），直接覆盖 ----
    def fill_years(row):
        if pd.notna(row['property_years']):
            return row['property_years']
        otype = row['ownership_type']
        return type_mode_years.get(otype, global_mode)

    df_comm['property_years'] = df_comm.apply(fill_years, axis=1)

    # ---- 保存小区表（直接覆盖原文件） ----
    df_comm.to_csv(comm_file, index=False, encoding='utf-8-sig')
    print(f"   ✅ 小区表产权字段已清洗并覆盖: {comm_file}")

    # ============================================================
    # 第二部分：映射到房源表（直接覆盖原列）
    # ============================================================
    ownership_map = dict(zip(
        df_comm['community_id'].astype(str).str.strip(),
        df_comm['ownership_type']
    ))

    years_map = dict(zip(
        df_comm['community_id'].astype(str).str.strip(),
        df_comm['property_years']
    ))

    print(f"   📌 构建映射完成: {len(ownership_map)} 个小区")

    for cat in CATEGORIES:
        house_type = "二手房" if cat == "sale" else "租房"
        in_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_filled_property.csv"

        if not in_file.exists():
            print(f"⚠️ 房源文件不存在: {in_file}，跳过")
            continue

        df_house = pd.read_csv(in_file, encoding='utf-8-sig')

        if 'community_id' not in df_house.columns:
            print(f"   ⚠️ 房源表缺少 community_id，跳过 {house_type}")
            continue

        df_house['community_id'] = df_house['community_id'].astype(str).str.strip()

        # ---- 直接覆盖 ownership_type 列 ----
        df_house['ownership_type'] = df_house['community_id'].map(ownership_map).fillna('未知')

        # ---- 直接覆盖 property_years 列 ----
        df_house['property_years'] = df_house['community_id'].map(years_map).fillna(global_mode)

        # ---- 保存（新文件名，供步骤5使用） ----
        out_file = OUTPUT_DIR / city / cat / f"{city}{house_type}_房源信息_filled_property_ownership.csv"
        for col in ['house_id', 'community_id']:
            if col in df_house.columns:
                df_house[col] = df_house[col].astype(str)
        df_house.to_csv(out_file, index=False, encoding='utf-8-sig', quoting=1)

        print(f"   ✅ 房源表产权字段已覆盖，保存至: {out_file}")


def main():
    print("=" * 60)
    print("步骤4：产权类型与产权年限清洗")
    print("   操作: 直接覆盖原列 ownership_type 和 property_years")
    print("   流程: 清洗小区表 → 覆盖保存 → 映射到房源表 → 覆盖保存")
    print("=" * 60)

    for city in CITIES:
        process_city(city)

    print("\n✅ 步骤4完成")


if __name__ == "__main__":
    main()