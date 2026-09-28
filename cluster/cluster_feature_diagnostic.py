# cluster/郭沛祺-cluster_feature_diagnostic.py
# 聚类前特征检验：数值相关性、类别关联性、类别-数值区分度
# 读取 _complete.csv（包含所有数值和类别特征）
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency, f_oneway
import warnings
warnings.filterwarnings('ignore')

# ================== 配置 ==================
OUTPUT_DIR = Path(__file__).parent / "feature_diagnostic_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).parent.parent
MODELING_DIR = BASE_DIR / "feature_engineering" / "modeling_data"

# ================== 特征定义 ==================
# 四个数值特征
NUMERIC_COLS = ['unit_price', 'house_age', 'rent_sale_ratio', 'total_amenity_score']

# 六个类别特征（初选）
CATEGORICAL_COLS = [
    'property_type',      # 物业类型
    'decoration',         # 装修程度
    'floor_type',         # 楼层类型
    'district',           # 行政区
    'ownership_type',     # 产权类型
    'orientation'         # 朝向
]

# ================== 辅助函数 ==================
def cramers_v(x, y):
    """计算两个分类变量的 Cramér's V 系数"""
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty:
        return 0
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    if n == 0:
        return 0
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    return np.sqrt(phi2corr / denom) if denom > 0 else 0

def cramers_v_matrix(df, cat_cols):
    """计算类别特征之间的 Cramér's V 矩阵"""
    n = len(cat_cols)
    v_matrix = pd.DataFrame(np.zeros((n, n)), index=cat_cols, columns=cat_cols)
    for i in range(n):
        for j in range(n):
            if i == j:
                v_matrix.iloc[i, j] = 1.0
            else:
                v_matrix.iloc[i, j] = cramers_v(df[cat_cols[i]], df[cat_cols[j]])
    return v_matrix

def anova_p_values(df, cat_cols, num_cols):
    """计算每个类别特征对每个数值特征的 ANOVA p 值"""
    results = []
    for cat in cat_cols:
        if df[cat].nunique() < 2:
            results.append({
                '类别特征': cat,
                '数值特征': '-',
                'p值': np.nan,
                '显著性': '-'
            })
            continue
        for num in num_cols:
            groups = []
            for val in df[cat].unique():
                data = df[df[cat] == val][num].dropna()
                if len(data) > 0:
                    groups.append(data)
            if len(groups) < 2:
                p_val = np.nan
            else:
                try:
                    f_stat, p_val = f_oneway(*groups)
                except:
                    p_val = np.nan
            if pd.notna(p_val):
                sig = '显著' if p_val < 0.05 else '不显著'
            else:
                sig = '-'
            results.append({
                '类别特征': cat,
                '数值特征': num,
                'p值': p_val,
                '显著性': sig
            })
    return pd.DataFrame(results)

def diagnose_group(df, group_name):
    """对单个组合进行特征诊断，保存详细指标"""
    print(f"\n{'='*50}")
    print(f"📊 诊断: {group_name}")
    print(f"{'='*50}")
    print(f"样本数: {len(df)}")

    # ---- 检查特征是否存在 ----
    missing_num = [c for c in NUMERIC_COLS if c not in df.columns]
    missing_cat = [c for c in CATEGORICAL_COLS if c not in df.columns]
    if missing_num or missing_cat:
        print(f"⚠️ 缺少特征: {missing_num + missing_cat}")
        num_cols = [c for c in NUMERIC_COLS if c in df.columns]
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
        if not num_cols or not cat_cols:
            print("❌ 关键特征缺失，跳过")
            return None
    else:
        num_cols = NUMERIC_COLS
        cat_cols = CATEGORICAL_COLS

    print(f"   ✅ 数值特征 ({len(num_cols)}): {num_cols}")
    print(f"   ✅ 类别特征 ({len(cat_cols)}): {cat_cols}")

    # ---- 1. 数值特征相关性矩阵 ----
    numeric_df = df[num_cols].dropna()
    if len(numeric_df) >= 2:
        corr_matrix = numeric_df.corr()
        print("\n--- 数值特征相关系数矩阵 ---")
        print(corr_matrix.round(3))
    else:
        corr_matrix = pd.DataFrame(index=num_cols, columns=num_cols)

    # ---- 2. 类别特征 Cramér's V 矩阵 ----
    cat_df = df[cat_cols].dropna()
    if len(cat_df) >= 2:
        v_matrix = cramers_v_matrix(cat_df, cat_cols)
        print("\n--- 类别特征 Cramér's V 矩阵 ---")
        print(v_matrix.round(3))
    else:
        v_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)

    # ---- 3. 类别-数值 ANOVA p 值表 ----
    anova_df = anova_p_values(df, cat_cols, num_cols)
    print("\n--- 类别-数值 ANOVA p 值 ---")
    for cat in cat_cols:
        sub = anova_df[anova_df['类别特征'] == cat]
        for _, row in sub.iterrows():
            if pd.notna(row['p值']):
                print(f"   {row['类别特征']} -> {row['数值特征']}: p={row['p值']:.4f} ({row['显著性']})")
            else:
                print(f"   {row['类别特征']} -> {row['数值特征']}: N/A")

    # ---- 保存结果 ----
    out_dir = OUTPUT_DIR / group_name
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_matrix.to_csv(out_dir / '数值相关系数矩阵.csv', encoding='utf-8-sig')
    v_matrix.to_csv(out_dir / 'CramersV关联矩阵.csv', encoding='utf-8-sig')
    anova_df.to_csv(out_dir / 'ANOVA_p值表.csv', index=False, encoding='utf-8-sig')

    print(f"\n   ✅ 诊断结果已保存至: {out_dir}")

    # 计算最大相关系数（绝对值）
    max_corr = 0
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            val = abs(corr_matrix.iloc[i, j])
            if val > max_corr:
                max_corr = val

    # 计算最大 Cramér's V
    max_v = 0
    for i in range(len(v_matrix.columns)):
        for j in range(i+1, len(v_matrix.columns)):
            val = abs(v_matrix.iloc[i, j])
            if val > max_v:
                max_v = val

    # 计算 ANOVA 不显著比例
    anova_sig = anova_df[anova_df['显著性'] != '-']
    if len(anova_sig) > 0:
        not_sig_ratio = (anova_sig[anova_sig['显著性'] == '不显著'].shape[0]) / anova_sig.shape[0]
    else:
        not_sig_ratio = 0

    return {
        '分组': group_name,
        '样本数': len(df),
        '数值特征最大|r|': max_corr,
        '类别特征最大CramérsV': max_v,
        'ANOVA不显著比例': not_sig_ratio
    }


def main():
    print("="*60)
    print("🔍 聚类前特征诊断")
    print(f"   数值特征 ({len(NUMERIC_COLS)}个): {NUMERIC_COLS}")
    print(f"   类别特征 ({len(CATEGORICAL_COLS)}个): {CATEGORICAL_COLS}")
    print(f"   初选特征共 {len(NUMERIC_COLS) + len(CATEGORICAL_COLS)} 维")
    print(f"   数据来源: _complete.csv (包含全部特征)")
    print(f"   输出目录: {OUTPUT_DIR}")
    print("="*60)

    if not MODELING_DIR.exists():
        print(f"❌ 建模数据目录不存在: {MODELING_DIR}")
        return

    groups = [d for d in MODELING_DIR.iterdir() if d.is_dir()]
    if not groups:
        print("❌ 未找到建模数据目录")
        return

    print(f"\n📌 找到 {len(groups)} 个分组: {[g.name for g in groups]}")

    all_summaries = []
    for group_dir in groups:
        group_name = group_dir.name
        # 读取 _complete.csv（包含全部特征）
        complete_file = group_dir / f"{group_name}_complete.csv"
        if not complete_file.exists():
            print(f"⚠️ 跳过 {group_name}：数据文件不存在 ({complete_file})")
            continue
        df = pd.read_csv(complete_file, encoding='utf-8-sig')
        print(f"📂 加载 {group_name}: {len(df)} 条")

        summary = diagnose_group(df, group_name)
        if summary:
            all_summaries.append(summary)

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(OUTPUT_DIR / '特征诊断汇总.csv', index=False, encoding='utf-8-sig')
        print("\n" + "="*60)
        print("📊 特征诊断汇总")
        print("="*60)
        print(summary_df.to_string(index=False))
        print(f"\n✅ 诊断报告已保存至: {OUTPUT_DIR / '特征诊断汇总.csv'}")
    else:
        print("❌ 无有效诊断数据")

if __name__ == "__main__":
    main()