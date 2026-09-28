# 相关性分析.py
# 功能：全国及分城市数值变量相关性热力图、分类变量ANOVA、城市间差异检验（市场分化）
# 修正：二手房与租房样本分别进行城市间差异检验，输出完整的Tukey HSD事后检验结果
# 新增：每个热力图同时保存对应的相关系数矩阵CSV
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import f_oneway, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

warnings.filterwarnings('ignore')

# ================== 配置 ==================
BASE_DIR = Path(__file__).parent.parent
FE_DIR = BASE_DIR / "feature_engineering" / "modeling_data"
OUTPUT_DIR = BASE_DIR / "EDA" / "results" / "相关性"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================== 中文字体设置 ==================
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
plt.rcParams['font.size'] = 14


def load_data():
    """从特征工程输出加载各城市各类型的完整合并表"""
    all_df = []
    for city in ["杭州", "金华", "临沂"]:
        for house_type in ["二手房", "租房"]:
            sub_dir = FE_DIR / f"{city}_{house_type}"
            complete_file = sub_dir / f"{city}_{house_type}_complete.csv"
            if complete_file.exists():
                df = pd.read_csv(complete_file, encoding='utf-8-sig')
                all_df.append(df)
                print(f"📂 加载 {city} {house_type}: {len(df)} 条")
            else:
                print(f"⚠️ 未找到 {city}_{house_type}_complete.csv")
    if not all_df:
        print("❌ 未找到特征工程的完整数据，请先运行郭沛祺-feature_engineering.py")
        return None
    df = pd.concat(all_df, ignore_index=True)
    numeric_cols = ['total_price', 'area', 'build_year', 'house_age',
                    'rent_sale_ratio', 'total_amenity_score',
                    'score_metro', 'score_school', 'score_hospital',
                    'score_shopping', 'score_dining']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"📂 总计加载完整数据: {len(df)} 条")
    return df


def plot_correlation_heatmap(df, house_type, output_path):
    """绘制热力图，并保存相关系数矩阵CSV"""
    numeric_cols = ['total_price', 'area', 'house_age',
                    'rent_sale_ratio', 'total_amenity_score',
                    'score_metro', 'score_school', 'score_hospital',
                    'score_shopping', 'score_dining']
    existing_cols = [col for col in numeric_cols if col in df.columns]
    if len(existing_cols) < 2:
        print(f"⚠️ {output_path} 可用的数值列不足，跳过")
        return
    numeric_df = df[existing_cols].dropna()
    if numeric_df.empty:
        print(f"⚠️ {output_path} 无可用的数值数据，跳过")
        return
    corr = numeric_df.corr()

    # 保存相关系数矩阵为CSV（与热力图同目录同名，扩展名为.csv）
    csv_path = output_path.with_suffix('.csv')
    corr.to_csv(csv_path, encoding='utf-8-sig')
    print(f"✅ 保存相关系数矩阵: {csv_path}")

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, ax=ax,
                annot_kws={'size': 10}, cbar_kws={'shrink': 0.8})
    ax.set_title(f'{house_type} 相关性矩阵', fontsize=16)
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存热力图: {output_path}")


def anova_categorical_vs_price(df, house_type, group_name):
    """单因素ANOVA：分类变量对价格的影响"""
    categorical_cols = ['decoration', 'elevator', 'floor_type', 'property_type',
                        'orientation', 'ownership_type', 'property_years']
    results = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        sub = df[[col, 'total_price']].dropna()
        if sub[col].nunique() < 2 or len(sub) < 10:
            continue
        try:
            model = ols(f'total_price ~ C({col})', data=sub).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            f_val = anova_table['F'][0]
            p_val = anova_table['PR(>F)'][0]
            ss_between = anova_table['sum_sq'][0]
            ss_total = ss_between + anova_table['sum_sq'][1]
            eta_sq = ss_between / ss_total
            results.append({
                '分组': group_name,
                '房源类型': house_type,
                '变量': col,
                'F值': f_val,
                'p值': p_val,
                '效应量_eta_sq': eta_sq,
                '有效样本数': len(sub)
            })
        except Exception as e:
            print(f"   ⚠️ {group_name} - {house_type} - {col} ANOVA 失败: {e}")
    return pd.DataFrame(results)


# ================== 城市间差异检验（分二手房/租房） ==================
def test_city_differences(df, house_type, diff_output_dir):
    """
    对指定房源类型进行城市间差异检验
    house_type: '二手房' 或 '租房'
    输出完整的Tukey HSD事后检验结果
    """
    print(f"\n--- {house_type} 城市间差异检验 ---")
    sub_df = df[df['house_type'] == house_type].copy()
    if sub_df.empty:
        print(f"⚠️ 无 {house_type} 数据，跳过")
        return

    sub_dir = diff_output_dir / house_type
    sub_dir.mkdir(parents=True, exist_ok=True)

    # ---- 连续变量 ----
    if house_type == '二手房':
        cont_vars = ['total_price', 'area', 'house_age', 'rent_sale_ratio', 'total_amenity_score']
    else:
        cont_vars = ['total_price', 'area', 'house_age', 'total_amenity_score']

    cont_vars = [var for var in cont_vars if var in sub_df.columns]

    # ANOVA汇总结果
    results_cont = []
    # Tukey详细结果（每个变量一个表）
    all_tukey_results = []

    for var in cont_vars:
        # 按城市分组
        groups = [group.dropna().values for name, group in sub_df.groupby('city')[var]]
        if len(groups) < 2:
            continue
        if all(len(g) < 3 for g in groups):
            continue
        try:
            # ANOVA
            f_stat, p_val = f_oneway(*groups)
            tukey_summary = None

            # 如果显著，做Tukey HSD
            if p_val < 0.05:
                data_for_tukey = sub_df[['city', var]].dropna()
                tukey = pairwise_tukeyhsd(endog=data_for_tukey[var],
                                          groups=data_for_tukey['city'],
                                          alpha=0.05)
                # 将Tukey结果转为DataFrame
                tukey_df = pd.DataFrame(
                    data=tukey.summary().data[1:],
                    columns=tukey.summary().data[0]
                )
                # 添加变量标识列
                tukey_df['变量'] = var
                tukey_df['房源类型'] = house_type
                all_tukey_results.append(tukey_df)

                # 生成显著配对摘要
                sig_df = tukey_df[tukey_df['reject'] == True]
                if not sig_df.empty:
                    tukey_summary = '; '.join([f"{row['group1']} vs {row['group2']}"
                                               for _, row in sig_df.iterrows()])
                else:
                    tukey_summary = '无显著配对'

            results_cont.append({
                '变量': var,
                'F统计量': f_stat,
                'p值': p_val,
                '显著性': '是' if p_val < 0.05 else '否',
                '显著差异配对': tukey_summary if tukey_summary else '无显著差异'
            })
        except Exception as e:
            print(f"   ⚠️ {var} 检验失败: {e}")

    # 保存ANOVA汇总
    if results_cont:
        cont_diff_df = pd.DataFrame(results_cont)
        cont_diff_df.to_csv(sub_dir / f'{house_type}_连续变量_城市间ANOVA差异.csv',
                            index=False, encoding='utf-8-sig')
        print(f"✅ 保存: {house_type}_连续变量_城市间ANOVA差异.csv")

    # 保存所有Tukey详细结果
    if all_tukey_results:
        combined_tukey = pd.concat(all_tukey_results, ignore_index=True)
        # 重新排列列的顺序，方便阅读
        cols = ['变量', '房源类型', 'group1', 'group2', 'meandiff', 'p-adj',
                'lower', 'upper', 'reject']
        existing_cols = [c for c in cols if c in combined_tukey.columns]
        combined_tukey = combined_tukey[existing_cols]
        combined_tukey.to_csv(sub_dir / f'{house_type}_连续变量_Tukey事后检验详细.csv',
                              index=False, encoding='utf-8-sig')
        print(f"✅ 保存: {house_type}_连续变量_Tukey事后检验详细.csv")

    # ---- 分类变量：卡方检验 ----
    cat_vars = ['property_type', 'decoration', 'floor_type', 'orientation', 'ownership_type']
    cat_vars = [var for var in cat_vars if var in sub_df.columns]

    results_cat = []
    for var in cat_vars:
        contingency = pd.crosstab(sub_df['city'], sub_df[var])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        try:
            chi2, p, dof, expected = chi2_contingency(contingency)
            results_cat.append({
                '变量': var,
                '卡方统计量': chi2,
                '自由度': dof,
                'p值': p,
                '显著性': '是' if p < 0.05 else '否'
            })
        except Exception as e:
            print(f"   ⚠️ {var} 卡方检验失败: {e}")

    if results_cat:
        cat_diff_df = pd.DataFrame(results_cat)
        cat_diff_df.to_csv(sub_dir / f'{house_type}_分类变量_城市间卡方检验.csv',
                           index=False, encoding='utf-8-sig')
        print(f"✅ 保存: {house_type}_分类变量_城市间卡方检验.csv")


def main():
    print("🚀 开始相关性分析...")
    df = load_data()
    if df is None:
        return

    all_anova_results = []

    # ----- 1. 全国汇总 -----
    print("\n" + "=" * 60)
    print("📊 全国汇总分析（三城合并）")
    print("=" * 60)
    for house_type in ['二手房', '租房']:
        sub = df[df['house_type'] == house_type]
        if sub.empty:
            continue
        heatmap_path = OUTPUT_DIR / f'全国_{house_type}_相关性热力图.png'
        plot_correlation_heatmap(sub, house_type, heatmap_path)
        anova_df = anova_categorical_vs_price(sub, house_type, '全国')
        if not anova_df.empty:
            all_anova_results.append(anova_df)
            anova_df.to_csv(OUTPUT_DIR / f'全国_{house_type}_类别变量ANOVA.csv',
                            index=False, encoding='utf-8-sig')
            print(f"✅ 保存: 全国_{house_type}_类别变量ANOVA.csv")

    # ----- 2. 分城市分析（简化路径）-----
    print("\n" + "=" * 60)
    print("📊 分城市分析")
    print("=" * 60)
    for city in df['city'].unique():
        city_df = df[df['city'] == city]
        print(f"\n--- 城市: {city} ---")
        city_out_dir = OUTPUT_DIR / city
        city_out_dir.mkdir(parents=True, exist_ok=True)
        for house_type in ['二手房', '租房']:
            sub = city_df[city_df['house_type'] == house_type]
            if sub.empty:
                continue
            heatmap_path = city_out_dir / f'{city}_{house_type}_相关性热力图.png'
            plot_correlation_heatmap(sub, house_type, heatmap_path)
            anova_df = anova_categorical_vs_price(sub, house_type, city)
            if not anova_df.empty:
                all_anova_results.append(anova_df)
                anova_df.to_csv(city_out_dir / f'{city}_{house_type}_类别变量ANOVA.csv',
                                index=False, encoding='utf-8-sig')
                print(f"✅ 保存: {city}_{house_type}_类别变量ANOVA.csv")

    # ----- 汇总ANOVA结果 -----
    if all_anova_results:
        combined = pd.concat(all_anova_results, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / '所有分析_类别变量ANOVA汇总.csv',
                        index=False, encoding='utf-8-sig')
        print("\n✅ 保存: 所有分析_类别变量ANOVA汇总.csv")

    # ============================================================
    # 城市间差异检验（市场分化信号，含完整Tukey事后检验）
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 城市间差异检验（市场分化信号）")
    print("   注：二手房与租房样本分别检验，输出完整Tukey HSD事后检验")
    print("=" * 60)

    diff_output_dir = OUTPUT_DIR / "城市间差异"
    diff_output_dir.mkdir(parents=True, exist_ok=True)

    test_city_differences(df, '二手房', diff_output_dir)
    test_city_differences(df, '租房', diff_output_dir)

    print(f"\n✅ 相关性分析完成，所有结果已保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()