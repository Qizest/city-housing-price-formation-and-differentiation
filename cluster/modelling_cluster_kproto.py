# 郭沛祺-modelling_cluster_kproto.py
# K-Prototypes 聚类分析
# 最终特征集（诊断后确定）：
#   数值: unit_price, house_age, rent_sale_ratio, total_amenity_score
#   类别: property_type, decoration, floor_type, district, orientation
# 数据来源: _complete.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes
import warnings
import os

warnings.filterwarnings('ignore')

# ================== 配置 ==================
BASE_DIR = Path(__file__).parent.parent
MODELING_DIR = BASE_DIR / "feature_engineering" / "modeling_data"
OUTPUT_DIR = Path(__file__).parent / "kproto_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================== 最终特征集 ==================
NUMERIC_COLS = ['unit_price', 'house_age', 'rent_sale_ratio', 'total_amenity_score']
CATEGORICAL_COLS = ['property_type', 'decoration', 'floor_type', 'district', 'orientation']

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
plt.rcParams['font.size'] = 12


# ================== 辅助函数 ==================
def determine_optimal_k(X_numeric, X_mixed, categorical_indices, k_range=(2, 8)):
    """使用肘部法和轮廓系数确定最优 K 值"""
    inertias = []
    silhouette_scores = []
    k_values = list(range(k_range[0], k_range[1] + 1))

    for k in k_values:
        print(f"   ⏳ 测试 K={k} ...")
        kproto = KPrototypes(n_clusters=k, init='Cao', n_init=3, max_iter=50, random_state=42, verbose=0)
        labels = kproto.fit_predict(X_mixed, categorical=categorical_indices)
        inertias.append(kproto.cost_)
        if k > 1 and X_numeric.shape[1] > 0:
            try:
                score = silhouette_score(X_numeric, labels)
                silhouette_scores.append(score)
            except:
                silhouette_scores.append(0)
        else:
            silhouette_scores.append(0)

    if max(silhouette_scores) > 0.1:
        optimal_k = k_values[np.argmax(silhouette_scores)]
    else:
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        if len(diffs2) > 0:
            optimal_k = k_values[np.argmax(diffs2) + 1]
        else:
            optimal_k = 3

    return optimal_k, inertias, silhouette_scores


def plot_k_selection(k_values, inertias, silhouette_scores, optimal_k, group_name, output_dir):
    """绘制肘部图和轮廓系数图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(k_values, inertias, 'bo-')
    ax1.axvline(x=optimal_k, color='red', linestyle='--', label=f'最优 K={optimal_k}')
    ax1.set_xlabel('K 值')
    ax1.set_ylabel('代价 (SSE)')
    ax1.set_title(f'{group_name} 肘部图')
    ax1.legend()
    ax1.grid(True)

    ax2 = axes[1]
    valid_k = k_values[1:] if len(silhouette_scores) >= len(k_values) else k_values
    valid_scores = silhouette_scores[1:] if len(silhouette_scores) >= len(k_values) else silhouette_scores
    ax2.plot(valid_k, valid_scores, 'go-')
    ax2.axvline(x=optimal_k, color='red', linestyle='--', label=f'最优 K={optimal_k}')
    ax2.set_xlabel('K 值')
    ax2.set_ylabel('轮廓系数')
    ax2.set_title(f'{group_name} 轮廓系数图')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f'{group_name}_K值选择.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_pca_cluster(X_numeric, labels, group_name, output_dir):
    """PCA 可视化聚类结果"""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_numeric)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=labels, cmap='Set1', s=25, alpha=0.7,
                         edgecolors='black', linewidths=0.3)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'{group_name} K-Prototypes PCA 可视化')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{group_name}_PCA散点图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存 PCA 散点图")


def process_house_clustering_kproto(df_house, city, house_type):
    """K-Prototypes 主流程"""
    group_name = f"{city}_{house_type}"
    print(f"\n{'=' * 50}")
    print(f"🏠 K-Prototypes: {group_name}")
    print(f"{'=' * 50}")

    # ---- 使用最终特征集 ----
    numeric_existing = [col for col in NUMERIC_COLS if col in df_house.columns]
    categorical_existing = [col for col in CATEGORICAL_COLS if col in df_house.columns]

    print(f"   📊 数值特征: {numeric_existing}")
    print(f"   📊 类别特征: {categorical_existing}")
    print(f"   📊 共 {len(numeric_existing) + len(categorical_existing)} 个特征参与聚类")

    if len(numeric_existing) < 2 or len(categorical_existing) < 2:
        print("⚠️ 特征不足，跳过")
        return None

    # ---- 数据预处理 ----
    df_sub = df_house[numeric_existing + categorical_existing].copy()
    for col in numeric_existing:
        df_sub[col] = df_sub[col].fillna(df_sub[col].median())
    for col in categorical_existing:
        df_sub[col] = df_sub[col].fillna('未知').astype(str)

    # ---- 标准化数值列 ----
    scaler = StandardScaler()
    df_sub[numeric_existing] = scaler.fit_transform(df_sub[numeric_existing])

    X_numeric = df_sub[numeric_existing].values.astype(np.float64)
    X_categorical = df_sub[categorical_existing].values.astype(str)
    X_mixed = np.hstack([X_numeric, X_categorical])
    categorical_indices = list(range(len(numeric_existing), X_mixed.shape[1]))

    # ---- 确定最优 K ----
    optimal_k, inertias, silhouette_scores = determine_optimal_k(X_numeric, X_mixed, categorical_indices)
    print(f"   ✅ 最优 K = {optimal_k}")

    # ---- 最终聚类 ----
    kproto = KPrototypes(n_clusters=optimal_k, init='Cao', n_init=5, max_iter=100, random_state=42, verbose=0)
    cluster_labels = kproto.fit_predict(X_mixed, categorical=categorical_indices)

    # ---- 输出目录 ----
    out_dir = OUTPUT_DIR / group_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 绘图 ----
    k_values = list(range(2, 9))
    plot_k_selection(k_values, inertias, silhouette_scores, optimal_k, group_name, out_dir)
    plot_pca_cluster(X_numeric, cluster_labels, group_name, out_dir)

    # ---- 簇中心解读 ----
    df_sub['cluster'] = cluster_labels
    numeric_means = df_sub.groupby('cluster')[numeric_existing].mean().round(3)
    categorical_modes = df_sub.groupby('cluster')[categorical_existing].agg(lambda x: x.mode()[0] if not x.mode().empty else '未知')

    # ---- 生成簇解读表 ----
    interpretations = []
    for i in sorted(numeric_means.index):
        size = df_sub['cluster'].value_counts().loc[i]
        means = numeric_means.loc[i]
        modes = categorical_modes.loc[i]

        unit_price = means.get('unit_price', 0)
        rent_ratio = means.get('rent_sale_ratio', 0)
        amenity = means.get('total_amenity_score', 0)
        house_age = means.get('house_age', 0)

        if unit_price > 0.5 and house_age < -0.3:
            cluster_type = '高单价次新品质型'
        elif unit_price > 0.5 and house_age > 0.3:
            cluster_type = '高单价老旧型（区位溢价）'
        elif unit_price < -0.5 and house_age < -0.3:
            cluster_type = '低单价次新实惠型'
        elif unit_price < -0.5 and house_age > 0.3:
            cluster_type = '低单价老旧型（价格洼地）'
        elif unit_price > 0.3:
            cluster_type = '中高单价型'
        elif unit_price < -0.3:
            cluster_type = '中低单价型'
        else:
            cluster_type = '均价均衡型'

        if rent_ratio > 0.5:
            rent_desc = '出租占比高'
        elif rent_ratio < -0.5:
            rent_desc = '出售占比高'
        else:
            rent_desc = '租售均衡'

        if amenity > 0.5:
            amenity_desc = '配套完善'
        elif amenity < -0.5:
            amenity_desc = '配套薄弱'
        else:
            amenity_desc = '配套适中'

        interpretations.append({
            '簇': i,
            '样本数': size,
            '占比': f"{size / len(df_sub) * 100:.1f}%",
            '类型': cluster_type,
            '租售比': rent_desc,
            '配套': amenity_desc,
            '主要行政区': modes.get('district', '未知'),
            '主要物业类型': modes.get('property_type', '未知'),
            '主要装修': modes.get('decoration', '未知'),
            '主要楼层': modes.get('floor_type', '未知'),
            '主要朝向': modes.get('orientation', '未知'),
            '单价(标准化)': f"{unit_price:.2f}",
            '房龄(标准化)': f"{house_age:.2f}"
        })

    interp_df = pd.DataFrame(interpretations)
    interp_df.to_csv(out_dir / f'{group_name}_簇解读表.csv', index=False, encoding='utf-8-sig')

    # ---- 保存聚类标签 ----
    df_house['cluster'] = cluster_labels
    df_house[['house_id', 'city', 'community_id', 'cluster']].to_csv(
        out_dir / f'{group_name}_聚类标签.csv', index=False, encoding='utf-8-sig'
    )

    print(f"✅ K-Prototypes 完成: {group_name} (K={optimal_k}, 样本数={len(df_sub)})")
    return {
        'group': group_name,
        'optimal_k': optimal_k,
        'n_samples': len(df_sub),
        'silhouette': max(silhouette_scores) if silhouette_scores else None
    }


def main():
    print("=" * 60)
    print("🚀 K-Prototypes 聚类分析（最终特征集）")
    print("   数据来源: _complete.csv")
    print("   数值特征: unit_price, house_age, rent_sale_ratio, total_amenity_score")
    print("   类别特征: property_type, decoration, floor_type, district, orientation")
    print("=" * 60)

    groups = [d for d in MODELING_DIR.iterdir() if d.is_dir()]
    if not groups:
        print("❌ 未找到建模数据目录，请先运行特征工程")
        return

    print(f"\n📌 找到 {len(groups)} 个组合:")
    for g in groups:
        print(f"   - {g.name}")

    summary = []
    for group_dir in groups:
        group_name = group_dir.name
        parts = group_name.split('_')
        if len(parts) != 2:
            continue
        city, house_type = parts[0], parts[1]

        complete_file = group_dir / f"{group_name}_complete.csv"
        if not complete_file.exists():
            print(f"⚠️ 跳过 {group_name}：数据文件不存在 ({complete_file})")
            continue

        df_house = pd.read_csv(complete_file, encoding='utf-8-sig')
        print(f"📂 加载 {group_name}: {len(df_house)} 条")

        result = process_house_clustering_kproto(df_house, city, house_type)
        if result:
            summary.append(result)

    print("\n" + "=" * 60)
    print("📊 K-Prototypes 聚类汇总")
    print("=" * 60)
    if summary:
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(OUTPUT_DIR / 'KPrototypes聚类汇总.csv', index=False, encoding='utf-8-sig')

    print(f"\n✅ K-Prototypes 聚类完成！结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()