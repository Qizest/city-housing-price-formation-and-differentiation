# regression/regression_diagnostic.py
# 回归诊断：多重共线性(VIF)、残差检验、低频率虚拟变量检查
# 只做诊断，不训练最终模型
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
MODELING_DIR = BASE_DIR / "regression" / "modeling_data"
OUTPUT_DIR = Path(__file__).parent / "regression_diagnostic_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 中文字体设置
font_paths = [
    'C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/msyh.ttc',
    '/System/Library/Fonts/PingFang.ttc',
]
font_path = None
for path in font_paths:
    if os.path.exists(path):
        font_path = path
        break

if font_path:
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12


def load_data(group_name):
    """加载训练集数据，正确处理布尔类型列"""
    group_dir = MODELING_DIR / group_name
    X_file = group_dir / f"{group_name}_X_train.csv"
    y_file = group_dir / f"{group_name}_y_train.csv"

    if not X_file.exists() or not y_file.exists():
        print(f"⚠️ 跳过 {group_name}：数据文件不完整")
        return None, None, None

    X = pd.read_csv(X_file, encoding='utf-8-sig')
    y = pd.read_csv(y_file, encoding='utf-8-sig').squeeze()

    # ---- 关键修复：将布尔列转换为整数 0/1 ----
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
        print(f"   🔧 将 {len(bool_cols)} 个布尔列转换为 0/1")

    # ---- 强制所有列为数值类型 ----
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # 将仍为 object 的列（通常不会发生）转换为 0
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = 0

    # 如果存在 total_price 列则删除（理论上不应该有）
    if 'total_price' in X.columns:
        X = X.drop(columns=['total_price'])

    # 删除含 NaN 的行
    valid = X.notna().all(axis=1) & y.notna()
    before = len(X)
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    if len(X) < before:
        print(f"   ⚠️ 删除了 {before - len(X)} 行含 NaN 的数据")

    parts = group_name.split('_')
    house_type = parts[1] if len(parts) > 1 else ''
    target_name = 'unit_price' if house_type == '二手房' else 'total_price'

    print(f"   ✅ 因变量: {target_name}")
    print(f"   ✅ 特征数: {len(X.columns)}")
    print(f"   ✅ 样本数: {len(X)}")
    return X, y, target_name


def fit_ols_model(X, y):
    """拟合 OLS 模型（仅用于获得残差进行诊断）"""
    # 确保 y 为正数
    y = np.asarray(y, dtype=np.float64)
    if (y <= 0).any():
        print("   ⚠️ 因变量包含非正值，将对所有值取绝对值并加1")
        y = np.abs(y) + 1e-6
    y_log = np.log(y)

    # 确保 X 全部为 float64
    X_arr = np.asarray(X, dtype=np.float64)
    X_const = sm.add_constant(X_arr)
    model = sm.OLS(y_log, X_const).fit(cov_type='HC3')
    return model, X_const


def calc_vif(X):
    """计算 VIF（仅对自变量，不含截距项）"""
    X_vif = X.copy()
    if 'const' in X_vif.columns:
        X_vif = X_vif.drop(columns=['const'])

    # 确保所有列是数值
    X_vif = X_vif.apply(pd.to_numeric, errors='coerce').fillna(0)

    vif_data = []
    for i, col in enumerate(X_vif.columns):
        if X_vif[col].nunique() <= 1:
            vif = np.inf
        else:
            try:
                vif = variance_inflation_factor(X_vif.values, i)
            except Exception:
                vif = np.inf
        vif_data.append({'变量': col, 'VIF': vif})
    return pd.DataFrame(vif_data)


def check_low_frequency_dummies(X, threshold=0.01):
    """检查 One-Hot 编码列中是否存在占比过低的水平"""
    low_freq_cols = []
    prefixes = ['district_', 'floor_type_', 'orientation_', 'decoration_', 'property_type_']
    for col in X.columns:
        if any(col.startswith(p) for p in prefixes):
            ratio = (X[col] == 1).mean()
            if ratio < threshold and ratio > 0:
                low_freq_cols.append({'变量': col, '占比': ratio})
    return pd.DataFrame(low_freq_cols)


def run_residual_tests(model, X, y):
    """执行残差诊断（正态性、异方差、自相关）"""
    X_const = sm.add_constant(X)
    resid = model.resid

    results = []

    # Jarque-Bera 正态性
    jb_stat, jb_p, _, _ = sm.stats.jarque_bera(resid)
    results.append({
        '检验名称': 'Jarque-Bera 正态性',
        '统计量': jb_stat,
        'p值': jb_p,
        '结论': '通过' if jb_p > 0.05 else '不通过'
    })

    # Shapiro-Wilk 正态性
    sample_resid = resid if len(resid) <= 5000 else np.random.choice(resid, 5000, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(sample_resid)
    results.append({
        '检验名称': 'Shapiro-Wilk 正态性',
        '统计量': shapiro_stat,
        'p值': shapiro_p,
        '结论': '通过' if shapiro_p > 0.05 else '不通过'
    })

    # Breusch-Pagan 同方差性
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(resid, X_const)
        results.append({
            '检验名称': 'Breusch-Pagan 同方差性',
            '统计量': bp_stat,
            'p值': bp_p,
            '结论': '通过' if bp_p > 0.05 else '不通过'
        })
    except Exception:
        results.append({
            '检验名称': 'Breusch-Pagan 同方差性',
            '统计量': np.nan,
            'p值': np.nan,
            '结论': '计算失败'
        })

    # Durbin-Watson 自相关
    dw_stat = durbin_watson(resid)
    dw_conclusion = '通过' if 1.5 < dw_stat < 2.5 else '警告'
    results.append({
        '检验名称': 'Durbin-Watson 自相关',
        '统计量': dw_stat,
        'p值': np.nan,
        '结论': dw_conclusion
    })

    # 条件数
    cond_num = np.linalg.cond(X_const)
    results.append({
        '检验名称': '条件数 (多重共线性)',
        '统计量': cond_num,
        'p值': np.nan,
        '结论': '正常' if cond_num < 100 else '警告'
    })

    return pd.DataFrame(results)


def process_group(group_name):
    """处理单个组合：执行诊断"""
    print(f"\n{'='*50}")
    print(f"🔍 诊断: {group_name}")
    print('='*50)

    X, y, target_name = load_data(group_name)
    if X is None:
        return None

    # ---- 1. 检查低频率虚拟变量 ----
    low_freq_df = check_low_frequency_dummies(X, threshold=0.01)
    if not low_freq_df.empty:
        print(f"   ⚠️ 发现 {len(low_freq_df)} 个低频率 One-Hot 列 (占比 < 1%)")
        low_list = low_freq_df['变量'].tolist()
        if len(low_list) > 5:
            print(f"      建议检查: {low_list[:5]}... (共{len(low_list)}个)")
        else:
            print(f"      建议检查: {low_list}")
    else:
        print("   ✅ 无低频率 One-Hot 列")

    # ---- 2. 拟合模型（仅用于诊断） ----
    model, X_const = fit_ols_model(X, y)
    print(f"   R²: {model.rsquared:.4f} (仅用于诊断参考，不代表最终模型)")

    # ---- 3. VIF 诊断 ----
    vif_df = calc_vif(X)
    vif_df['VIF_status'] = vif_df['VIF'].apply(
        lambda x: '高共线性' if x > 10 else ('中等' if x > 5 else '正常') if x != np.inf else '常量列'
    )
    high_vif = vif_df[vif_df['VIF'] > 10]
    if not high_vif.empty:
        print(f"   ⚠️ VIF > 10 的变量数量: {len(high_vif)}")
        high_list = high_vif['变量'].tolist()
        if len(high_list) > 5:
            print(f"      建议剔除: {high_list[:5]}... (共{len(high_list)}个)")
        else:
            print(f"      建议剔除: {high_list}")
    else:
        print("   ✅ 无高 VIF 变量")

    # ---- 4. 残差诊断 ----
    diag_df = run_residual_tests(model, X, y)
    print("\n--- 残差诊断 ---")
    for _, row in diag_df.iterrows():
        status = "✅" if row['结论'] in ['通过', '正常'] else "⚠️" if row['结论'] == '警告' else "❌"
        print(f"   {status} {row['检验名称']}: {row['结论']}")

    # ---- 5. 输出 ----
    out_dir = OUTPUT_DIR / group_name
    out_dir.mkdir(parents=True, exist_ok=True)

    vif_df.to_csv(out_dir / f'{group_name}_VIF.csv', index=False, encoding='utf-8-sig')
    low_freq_df.to_csv(out_dir / f'{group_name}_低频率虚拟变量.csv', index=False, encoding='utf-8-sig')
    diag_df.to_csv(out_dir / f'{group_name}_诊断结果.csv', index=False, encoding='utf-8-sig')

    print(f"\n   ✅ 诊断结果已保存至: {out_dir}")

    return {
        'group': group_name,
        'n_samples': len(X),
        'n_features': len(X.columns),
        'n_high_vif': len(high_vif),
        'n_low_freq': len(low_freq_df),
        'r2_diagnostic': model.rsquared
    }


def main():
    print("=" * 60)
    print("🔍 回归诊断 (VIF + 残差检验)")
    print("   数据来源: regression/modeling_data/*/")
    print("   VIF > 10 → 建议剔除")
    print("   低频率虚拟变量 (占比 < 1%) → 建议检查/合并")
    print("   此脚本只做诊断，不训练最终模型")
    print("   根据诊断结果调整变量后，由后续建模脚本训练最终模型")
    print("=" * 60)

    if not MODELING_DIR.exists():
        print(f"❌ 回归建模数据目录不存在: {MODELING_DIR}")
        print("   请先运行 regression_prepare.py")
        return

    groups = [d.name for d in MODELING_DIR.iterdir() if d.is_dir()]
    if not groups:
        print("❌ 未找到回归建模数据")
        return

    summary = []
    for group_name in groups:
        result = process_group(group_name)
        if result:
            summary.append(result)

    if summary:
        print("\n" + "=" * 60)
        print("📊 回归诊断汇总")
        print("=" * 60)
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(OUTPUT_DIR / '回归诊断汇总.csv', index=False, encoding='utf-8-sig')

    print(f"\n✅ 诊断完成！结果保存在: {OUTPUT_DIR}")
    print("   请根据诊断结果（VIF、低频率虚拟变量）调整回归变量后")
    print("   再运行最终建模脚本训练回归模型。")


if __name__ == "__main__":
    main()