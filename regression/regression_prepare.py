# regression/regression_prepare.py
# 回归建模数据准备：One-Hot 编码 + 训练/测试集划分
# 输入：feature_engineering/modeling_data/*/_complete.csv
# 输出：regression/modeling_data/*/ 下分别保存 X_train, X_test, y_train, y_test, metadata
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent.parent
FE_DIR = BASE_DIR / "feature_engineering" / "modeling_data"
OUTPUT_DIR = BASE_DIR / "regression" / "modeling_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.3

# 需要 One-Hot 编码的类别特征（不含 ownership_type，因其与 property_type 高度相关）
CATEGORICAL_COLS = ['district', 'floor_type', 'decoration', 'property_type', 'orientation']

# 二值变量（直接作为数值特征使用，不进行 One-Hot）
BINARY_COLS = ['elevator_encoded']

# 元数据列（不进入模型）
METADATA_COLS = ['house_id', 'community_id', 'community_name', 'city', 'house_type', 'location']


def load_data(group_name):
    """加载特征工程输出的完整数据"""
    group_dir = FE_DIR / group_name
    complete_file = group_dir / f"{group_name}_complete.csv"
    if not complete_file.exists():
        print(f"⚠️ 数据文件不存在: {complete_file}")
        return None
    df = pd.read_csv(complete_file, encoding='utf-8-sig')
    print(f"📂 加载 {group_name}: {len(df)} 条")
    return df


def prepare_regression_data(df, group_name):
    """为回归任务准备数据：One-Hot 编码 + 训练/测试集划分"""
    parts = group_name.split('_')
    city, house_type = parts[0], parts[1]
    is_sale = (house_type == '二手房')

    # ---- 确定因变量 ----
    if is_sale:
        target_col = 'unit_price'
        exclude_area = True
    else:
        target_col = 'total_price'
        exclude_area = False

    print(f"   📌 因变量: {target_col}")
    print(f"   📌 排除 area: {exclude_area}")

    # ---- 提取特征和因变量 ----
    metadata = df[METADATA_COLS].copy()

    # ============================================================
    # 明确排除的列
    # ============================================================
    exclude_cols = METADATA_COLS + [
        # ---- 文本/冗余字段 ----
        'layout',  # 户型描述，文本信息
        'floor',  # 原始楼层文本，已提取 floor_type
        'build_year',  # 建成年份，已转为 house_age
        'build_year_comm',  # 小区建成年份，与 house_age 高度相关
        'total_floor',  # 总楼层，与 floor_type 高度相关
        'property_years',  # 产权年限，与房龄高度相关
        'property_years_comm',  # 小区产权年限
        'ownership_type',  # 与 property_type 高度相关（Cramér's V ≈ 0.707）
        'elevator',  # 原始文本，已转为 elevator_encoded
        # ---- 小区聚合变量（回归中排除，避免数据泄漏） ----
        'price',  # 小区均价
        'nsh',  # 小区在售数量
        'nrp',  # 小区在租数量
        # ---- 聚类特征（回归中暂不纳入） ----
        'rent_sale_ratio',
        'total_amenity_score',
        'score_metro', 'score_school', 'score_hospital',
        'score_shopping', 'score_dining',
        # ⚠️ 注意：unit_price 不在此处排除！
        # 它作为二手房模型的因变量，由下面的 target_col 排除逻辑动态处理
        # 租房模型中因变量为 total_price，unit_price 会因共线性被后续 VIF 诊断剔除
    ]

    # 周边文本描述列（around_ 开头不含 _count）
    around_text = [col for col in df.columns if col.startswith('around_') and '_count' not in col]
    exclude_cols.extend(around_text)

    # 周边计数列（around_*_count）
    around_count = [col for col in df.columns if col.startswith('around_') and '_count' in col]
    exclude_cols.extend(around_count)

    # 二手房排除 area
    if exclude_area and 'area' in df.columns:
        exclude_cols.append('area')

    # ---- ⚠️ 关键：排除 target 本身 ----
    # 当 target_col = 'unit_price' 时，exclude_cols 包含 unit_price，不会被当作自变量
    # 当 target_col = 'total_price' 时，exclude_cols 包含 total_price，unit_price 保留
    # 但租房模型中 unit_price 仍可能被 VIF 诊断剔除
    exclude_cols.append(target_col)

    exclude_cols = [col for col in exclude_cols if col in df.columns]

    # ---- 提取数值特征 ----
    # 在 CATEGORICAL_COLS 和 BINARY_COLS 之外的数值列
    numeric_cols = [col for col in df.columns
                    if col not in exclude_cols
                    and col not in CATEGORICAL_COLS
                    and col not in BINARY_COLS
                    and pd.api.types.is_numeric_dtype(df[col])]

    X_numeric = df[numeric_cols].copy()
    for col in X_numeric.columns:
        if X_numeric[col].isna().any():
            median_val = X_numeric[col].median()
            if pd.isna(median_val):
                median_val = 0
            X_numeric[col] = X_numeric[col].fillna(median_val)

    # ---- 提取二值变量（elevator_encoded） ----
    binary_cols = [col for col in BINARY_COLS if col in df.columns and col not in exclude_cols]
    X_binary = df[binary_cols].copy()
    for col in X_binary.columns:
        if X_binary[col].isna().any():
            X_binary[col] = X_binary[col].fillna(0)

    # ---- One-Hot 编码类别特征 ----
    cat_cols = [col for col in CATEGORICAL_COLS if col in df.columns and col not in exclude_cols]
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True)

    # ---- 合并所有特征 ----
    X = pd.concat([X_numeric, X_binary, X_cat], axis=1)
    X = X.fillna(0)

    print(f"   📊 数值特征: {len(numeric_cols)} 个")
    print(f"   📊 二值变量: {len(binary_cols)} 个")
    print(f"   📊 类别特征: {len(cat_cols)} 个")
    print(f"   📊 总特征数: {len(X.columns)}")

    # ---- 提取因变量 ----
    y = df[target_col].copy()
    valid_mask = y.notna()
    if not valid_mask.all():
        print(f"   ⚠️ 因变量存在 {(~valid_mask).sum()} 个缺失值，将删除对应行")
        X = X[valid_mask]
        y = y[valid_mask]
        metadata = metadata[valid_mask]

    print(f"   📊 有效样本数: {len(X)}")

    # ---- 划分训练/测试集 ----
    X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
        X, y, metadata, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"   📊 训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")

    # ---- 保存 ----
    group_output_dir = OUTPUT_DIR / group_name
    group_output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(group_output_dir / f"{group_name}_X_train.csv", index=False, encoding='utf-8-sig')
    X_test.to_csv(group_output_dir / f"{group_name}_X_test.csv", index=False, encoding='utf-8-sig')
    y_train.to_csv(group_output_dir / f"{group_name}_y_train.csv", index=False, encoding='utf-8-sig')
    y_test.to_csv(group_output_dir / f"{group_name}_y_test.csv", index=False, encoding='utf-8-sig')
    metadata_train.to_csv(group_output_dir / f"{group_name}_metadata_train.csv", index=False, encoding='utf-8-sig')
    metadata_test.to_csv(group_output_dir / f"{group_name}_metadata_test.csv", index=False, encoding='utf-8-sig')

    pd.DataFrame({'feature': X.columns}).to_csv(
        group_output_dir / f"{group_name}_feature_cols.csv", index=False, encoding='utf-8-sig'
    )

    print(f"   ✅ 数据已保存至: {group_output_dir}")

    return {
        'group': group_name,
        'n_samples': len(X),
        'n_features': len(X.columns),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }


def main():
    print("=" * 60)
    print("🚀 回归建模数据准备")
    print("   输入: feature_engineering/modeling_data/*/_complete.csv")
    print("   输出: regression/modeling_data/*/")
    print("   One-Hot 编码 (drop_first=True)")
    print("   二值变量: elevator_encoded (直接保留)")
    print("   二手房因变量: unit_price (排除 area)")
    print("   租房因变量: total_price (保留 area)")
    print("   ⚠️ unit_price 作为二手房因变量被保留，租房模型中由 VIF 诊断决定是否剔除")
    print("   test_size = 0.3, random_state = 42")
    print("=" * 60)

    if not FE_DIR.exists():
        print(f"❌ 特征工程目录不存在: {FE_DIR}")
        return

    groups = [d.name for d in FE_DIR.iterdir() if d.is_dir()]
    if not groups:
        print("❌ 未找到数据分组")
        return

    print(f"\n📌 找到 {len(groups)} 个分组: {groups}")

    summary = []
    for group_name in groups:
        print(f"\n{'=' * 50}")
        print(f"处理: {group_name}")
        print('=' * 50)

        df = load_data(group_name)
        if df is None:
            continue

        result = prepare_regression_data(df, group_name)
        if result:
            summary.append(result)

    if summary:
        print("\n" + "=" * 60)
        print("📊 回归建模数据准备汇总")
        print("=" * 60)
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(OUTPUT_DIR / '回归建模数据准备汇总.csv', index=False, encoding='utf-8-sig')

    print(f"\n✅ 回归建模数据准备完成！结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()