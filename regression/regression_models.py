# regression/regression_models.py
# 分城市分类型回归建模（根据各组合诊断结果分别设计）
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELING_DIR = BASE_DIR / "regression" / "modeling_data"
OUTPUT_DIR = Path(__file__).parent / "regression_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_GROUPS = [
    '杭州_二手房', '杭州_租房',
    '金华_二手房', '金华_租房',
    '临沂_二手房', '临沂_租房'
]


def get_exclude_cols(group_name):
    """根据各组合诊断结果，返回需要剔除的特征列名"""
    is_rent = '租房' in group_name

    exclude = ['plot_ratio', 'greening_rate', 'elevator_encoded']

    if group_name == '杭州_二手房':
        exclude.extend(['district_建德', 'district_淳安'])
    elif group_name == '杭州_租房':
        exclude.extend(['district_桐庐', 'district_淳安', 'floor_type_地下', 'property_type_别墅'])
    elif group_name == '金华_二手房':
        exclude.extend(['district_磐安', 'floor_type_地下'])
    elif group_name == '金华_租房':
        exclude.extend(['district_武义县', 'district_浦江县', 'floor_type_地下'])
    elif group_name == '临沂_二手房':
        exclude.extend(['district_沂南', 'district_蒙阴', 'district_高新区', 'floor_type_地下'])
    elif group_name == '临沂_租房':
        exclude.extend(['district_沂南', 'district_蒙阴', 'district_高新区', 'floor_type_地下'])

    if is_rent:
        exclude.append('unit_price')

    return exclude


def process_decoration(X, group_name):
    """处理装修类别合并（仅租房模型）"""
    is_rent = '租房' in group_name
    if not is_rent:
        return X

    if 'decoration_精装修' in X.columns and 'decoration_简单装修' in X.columns:
        X['decoration_精装修'] = X.get('decoration_精装修', 0) + X.get('decoration_简单装修', 0)
        X = X.drop(columns=['decoration_简单装修'])

    if group_name == '金华_租房' and 'decoration_豪华装修' in X.columns:
        X['decoration_精装修'] = X.get('decoration_精装修', 0) + X.get('decoration_豪华装修', 0)
        X = X.drop(columns=['decoration_豪华装修'])

    return X


def prepare_features(X, group_name):
    """根据各组合诊断结果定制特征处理"""
    X_processed = X.copy()

    # 剔除朝向
    orient_cols = [col for col in X_processed.columns if col.startswith('orientation_')]
    if orient_cols:
        X_processed = X_processed.drop(columns=orient_cols)

    # 剔除指定列
    exclude_cols = get_exclude_cols(group_name)
    cols_to_drop = [col for col in exclude_cols if col in X_processed.columns]
    if cols_to_drop:
        X_processed = X_processed.drop(columns=cols_to_drop)

    # 装修合并
    X_processed = process_decoration(X_processed, group_name)

    # 强制数值类型
    for col in X_processed.columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0)

    return X_processed


def run_model(group_name):
    print(f"\n📊 {group_name}")

    group_dir = MODELING_DIR / group_name

    X_train = pd.read_csv(group_dir / f"{group_name}_X_train.csv", encoding='utf-8-sig')
    y_train = pd.read_csv(group_dir / f"{group_name}_y_train.csv", encoding='utf-8-sig').squeeze()
    X_test = pd.read_csv(group_dir / f"{group_name}_X_test.csv", encoding='utf-8-sig')
    y_test = pd.read_csv(group_dir / f"{group_name}_y_test.csv", encoding='utf-8-sig').squeeze()

    parts = group_name.split('_')
    house_type = parts[1]
    target_name = 'unit_price' if house_type == '二手房' else 'total_price'

    X_train_processed = prepare_features(X_train, group_name)
    X_test_processed = prepare_features(X_test, group_name)

    X_train_const = sm.add_constant(X_train_processed)
    X_test_const = sm.add_constant(X_test_processed)

    # 特征对齐
    train_cols = X_train_const.columns
    for col in train_cols:
        if col not in X_test_const.columns:
            X_test_const[col] = 0.0
    X_test_const = X_test_const[train_cols]

    X_train_const = X_train_const.astype(float)
    X_test_const = X_test_const.astype(float)

    print(f"   特征数: {len(X_train_const.columns) - 1} | 训练集: {len(X_train)} | 测试集: {len(X_test)}")

    y_train_log = np.log(y_train).astype(float)
    y_test_log = np.log(y_test).astype(float)

    model = sm.OLS(y_train_log, X_train_const).fit(cov_type='HC3')

    y_train_pred = model.predict(X_train_const)
    train_r2 = 1 - np.sum((y_train_log - y_train_pred) ** 2) / np.sum((y_train_log - np.mean(y_train_log)) ** 2)
    n_train = len(y_train)
    k_train = X_train_const.shape[1] - 1
    train_adj_r2 = 1 - (1 - train_r2) * (n_train - 1) / (n_train - k_train - 1)

    y_test_pred = model.predict(X_test_const)
    test_r2 = 1 - np.sum((y_test_log - y_test_pred) ** 2) / np.sum((y_test_log - np.mean(y_test_log)) ** 2)
    n_test = len(y_test)
    test_adj_r2 = 1 - (1 - test_r2) * (n_test - 1) / (n_test - k_train - 1)

    print(f"   R²: 训练集 {train_r2:.4f} | 测试集 {test_r2:.4f} (差距 {train_r2 - test_r2:.4f})")

    # 保存结果
    with open(OUTPUT_DIR / f"{group_name}_full_summary.txt", "w", encoding='utf-8') as f:
        f.write(model.summary().as_text())

    coef_df = pd.DataFrame({
        'variable': model.params.index,
        'coef': model.params.values,
        'std_err': model.bse.values,
        't': model.tvalues.values,
        'pvalue': model.pvalues.values,
        'signif': ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else '' for p in model.pvalues]
    })
    coef_df['pct_effect'] = (np.exp(coef_df['coef']) - 1) * 100
    coef_df.loc[coef_df['variable'] == 'const', 'pct_effect'] = np.nan
    coef_df.to_csv(OUTPUT_DIR / f"{group_name}_coefficients.csv", index=False, encoding='utf-8-sig')

    summary_df = pd.DataFrame({
        '指标': ['训练集R²', '训练集AdjR²', '测试集R²', '测试集AdjR²', '训练集样本数', '测试集样本数'],
        '值': [train_r2, train_adj_r2, test_r2, test_adj_r2, n_train, n_test]
    })
    summary_df.to_csv(OUTPUT_DIR / f"{group_name}_summary.csv", index=False, encoding='utf-8-sig')

    return {
        'group': group_name,
        'train_r2': train_r2,
        'train_adj_r2': train_adj_r2,
        'test_r2': test_r2,
        'test_adj_r2': test_adj_r2,
        'n_train': n_train,
        'n_test': n_test
    }


def main():
    print("=" * 60)
    print("🚀 分城市分类型回归建模")
    print("   通用剔除: plot_ratio, greening_rate, elevator_encoded, orientation_*")
    print("   租房: 简单装修→精装修 | 金华租房: 豪华装修→精装修")
    print("=" * 60)

    results = []
    for group_name in ALL_GROUPS:
        try:
            result = run_model(group_name)
            results.append(result)
        except Exception as e:
            print(f"   ❌ {group_name} 失败: {e}")

    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(OUTPUT_DIR / "所有模型汇总.csv", index=False, encoding='utf-8-sig')
        print("\n" + "=" * 60)
        print("📊 汇总")
        print("=" * 60)
        print(summary_df.to_string(index=False))

    print(f"\n✅ 完成: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()