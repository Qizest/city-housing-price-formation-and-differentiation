# feature_engineering.py
# 特征工程：构造新特征、剔除冗余、输出基础数据表（供EDA和建模使用）
# 输出：完整合并表（含派生特征）、房源独立表、小区独立表
# 不进行One-Hot编码，所有分类变量保留原始值，由下游建模脚本自行处理
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
HOUSE_FILE = BASE_DIR / "data_after_clean" / "完整房源数据.csv"
COMMUNITY_FILE = BASE_DIR / "data_after_clean" / "完整小区数据.csv"
OUTPUT_DIR = BASE_DIR / "feature_engineering" / "modeling_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_YEAR = 2026


def load_data():
    df_house = pd.read_csv(HOUSE_FILE, encoding='utf-8-sig')
    df_comm = pd.read_csv(COMMUNITY_FILE, encoding='utf-8-sig')
    print(f"📂 加载房源: {len(df_house)} 条")
    print(f"📂 加载小区: {len(df_comm)} 条")
    return df_house, df_comm


def merge_house_community(df_house, df_comm):
    df_house['community_id'] = df_house['community_id'].astype(str).str.strip()
    df_comm['community_id'] = df_comm['community_id'].astype(str).str.strip()
    df = df_house.merge(df_comm, on='community_id', how='left', suffixes=('', '_comm'))
    print(f"📊 关联后记录数: {len(df)}")
    return df


def derive_features(df):
    """派生新特征：房龄、单价、租售比、各配套单项得分及总分"""
    # 绿化率、容积率转数值
    for col in ['greening_rate', 'plot_ratio']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace('%', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 房龄（新增 house_age，保留 build_year 不变）
    if 'build_year' in df.columns:
        df['house_age'] = CURRENT_YEAR - df['build_year']
        df.loc[(df['house_age'] < 0) | (df['house_age'] > 200), 'house_age'] = np.nan

    # 单价
    if 'total_price' in df.columns and 'area' in df.columns:
        df['unit_price'] = df['total_price'] / df['area']
        df.loc[df['unit_price'] > 100000, 'unit_price'] = np.nan

    # 租售比
    if 'nrp' in df.columns and 'nsh' in df.columns:
        df['rent_sale_ratio'] = df['nrp'] / df['nsh'].replace(0, np.nan)

    # ---- 各配套单项得分 ----
    df['score_metro'] = np.log1p(df.get('around_地铁_count', 0)) * 3
    df['score_school'] = np.log1p(df.get('around_学校_count', 0)) * 2.5
    df['score_hospital'] = np.log1p(df.get('around_医院_count', 0)) * 2.0
    df['score_shopping'] = np.log1p(df.get('around_购物_count', 0)) * 1.6
    df['score_dining'] = np.log1p(df.get('around_餐饮_count', 0)) * 1.3

    # 总配套得分
    df['total_amenity_score'] = (
        df['score_metro'] + df['score_school'] + df['score_hospital'] +
        df['score_shopping'] + df['score_dining']
    )

    print("📊 派生特征完成")
    return df


def handle_missing(df):
    df_clean = df.copy()

    # 周边配套计数缺失填0
    count_cols = ['around_公交_count', 'around_地铁_count', 'around_学校_count',
                  'around_餐饮_count', 'around_购物_count', 'around_医院_count',
                  'around_银行_count']
    for col in count_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    # 连续型特征中位数填充
    continuous_cols = ['house_age', 'greening_rate', 'plot_ratio', 'price', 'nsh', 'nrp',
                       'rent_sale_ratio', 'total_amenity_score',
                       'score_metro', 'score_school', 'score_hospital', 'score_shopping', 'score_dining']
    for col in continuous_cols:
        if col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            median_val = df_clean[col].median()
            if pd.notna(median_val):
                df_clean[col] = df_clean[col].fillna(median_val)
            else:
                df_clean[col] = df_clean[col].fillna(0)

    # 分类列（保留原始值，不编码）
    categorical_cols = ['district', 'orientation', 'floor_type', 'decoration',
                        'property_type', 'ownership_type', 'elevator']
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('未知')

    # 对面积和总价进行3σ截尾
    for col in ['area', 'total_price']:
        if col in df_clean.columns:
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            if std_val > 0:
                upper_bound = mean_val + 3 * std_val
                lower_bound = max(0, mean_val - 3 * std_val)
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

    print(f"📊 缺失值处理完成")
    return df_clean


def process_group(df_group, city, house_type):
    group_name = f"{city}_{house_type}"
    print(f"\n{'=' * 50}")
    print(f"🏠 处理: {group_name} (样本数: {len(df_group)})")
    print(f"{'=' * 50}")

    group_output_dir = OUTPUT_DIR / group_name
    group_output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 派生特征 ----
    df = derive_features(df_group)

    # ---- 删除冗余文本列 ----
    cols_to_drop = ['house_title', 'page_tags', 'community_url', 'tags', 'tag']
    cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
    if cols_to_drop_existing:
        df = df.drop(columns=cols_to_drop_existing)
        print(f"   🗑️ 删除冗余字段: {', '.join(cols_to_drop_existing)}")

    # ---- 缺失值处理 ----
    df_clean = handle_missing(df)

    # ============================================================
    # 输出三个表：完整合并、房源独立、小区独立
    # ============================================================

    # 1. 完整合并表（所有字段，分类变量保留原始值）
    complete_file = group_output_dir / f"{group_name}_complete.csv"
    df_clean.to_csv(complete_file, index=False, encoding='utf-8-sig')
    print(f"✅ 保存完整合并表: {complete_file}")

    # 2. 房源独立表（供EDA绘图使用，保留原始字段 + 派生特征）
    house_base = [
        'house_id', 'community_id', 'city', 'house_type',
        'district', 'community_name', 'layout',
        'area', 'total_price', 'unit_price',
        'floor', 'floor_type',
        'orientation', 'decoration', 'build_year', 'house_age',
        'elevator', 'property_type', 'ownership_type', 'property_years'
    ]
    # 周边配套描述列（around_ 文本，不含 _count）
    around_text = [col for col in df_clean.columns if col.startswith('around_') and '_count' not in col]
    house_cols = [col for col in house_base + around_text if col in df_clean.columns]
    house_df = df_clean[house_cols].drop_duplicates(subset=['house_id'], keep='first')
    house_file = group_output_dir / f"{group_name}_house.csv"
    house_df.to_csv(house_file, index=False, encoding='utf-8-sig')
    print(f"✅ 保存房源独立表: {house_file} ({len(house_df)} 条)")

    # 3. 小区独立表（供EDA绘图使用，保留原始字段 + 派生特征）
    comm_base = [
        'community_id', 'community_name', 'city',
        'price', 'build_year',
        'nsh', 'nrp',
        'greening_rate', 'plot_ratio',
        'ownership_type', 'property_years',
        'property_type',
        'rent_sale_ratio',
        'score_metro', 'score_school', 'score_hospital', 'score_shopping', 'score_dining',
        'total_amenity_score'
    ]
    # 周边配套计数列（around_*_count）
    around_counts = [col for col in df_clean.columns if col.startswith('around_') and '_count' in col]
    tag_col = ['tag'] if 'tag' in df_clean.columns else []
    comm_cols = [col for col in comm_base + around_counts + tag_col if col in df_clean.columns]
    community_df = df_clean[comm_cols].drop_duplicates(subset=['community_id'], keep='first')
    community_file = group_output_dir / f"{group_name}_community.csv"
    community_df.to_csv(community_file, index=False, encoding='utf-8-sig')
    print(f"✅ 保存小区独立表: {community_file} ({len(community_df)} 个)")

    return {
        'group': group_name,
        'city': city,
        'house_type': house_type,
        'n_samples': len(df_group)
    }


def main():
    print("=" * 60)
    print("🚀 开始特征工程（构造特征、剔除冗余、输出基础表）")
    print("   输出: 完整合并表 | 房源独立表 | 小区独立表")
    print("   分类变量保留原始值，不进行One-Hot编码")
    print("   后续聚类/回归脚本自行处理编码")
    print("=" * 60)

    df_house, df_comm = load_data()
    df = merge_house_community(df_house, df_comm)

    groups = df.groupby(['city', 'house_type']).size().reset_index(name='count')
    groups = groups[groups['count'] >= 30]
    print(f"\n📌 有效组合 ({len(groups)} 个):")
    print(groups.to_string(index=False))

    summary = []
    for _, row in groups.iterrows():
        city = row['city']
        house_type = row['house_type']
        df_group = df[(df['city'] == city) & (df['house_type'] == house_type)].copy()
        result = process_group(df_group, city, house_type)
        summary.append(result)

    print("\n" + "=" * 60)
    print("📊 特征工程汇总")
    print("=" * 60)
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    summary_file = OUTPUT_DIR / "特征工程汇总.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 汇总报告已保存: {summary_file}")

    print("\n✅ 特征工程完成！")


if __name__ == "__main__":
    main()