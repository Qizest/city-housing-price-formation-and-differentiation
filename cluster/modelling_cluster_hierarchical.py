# cluster/郭沛祺-modelling_cluster_hierarchical.py
# 层次聚类 + Gower 距离（全量数据）
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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
import os
import time

warnings.filterwarnings('ignore')

# ================== 配置 ==================
BASE_DIR = Path(__file__).parent.parent
MODELING_DIR = BASE_DIR / "feature_engineering" / "modeling_data"
OUTPUT_DIR = Path(__file__).parent / "hierarchical_results"
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
sns.set_style("whitegrid", rc={'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'PingFang SC']})


def compute_gower_distance(df, numeric_cols, categorical_cols):
    """计算 Gower 距离矩阵（全量数据）"""
    n = len(df)
    print(f"   ⏳ 计算 {n}x{n} Gower 距离矩阵...")
    start_time = time.time()

    # 数值特征：0-1 归一化
    numeric_data = df[numeric_cols].values.astype(np.float64)
    for i in range(numeric_data.shape[1]):
        col = numeric_data[:, i]
        if col.max() - col.min() > 0:
            numeric_data[:, i] = (col - col.min()) / (col.max() - col.min())
        else:
            numeric_data[:, i] = 0

    # 类别特征
    categorical_data = df[categorical_cols].values.astype(str)

    from sklearn.metrics.pairwise import euclidean_distances
    num_dist = euclidean_distances(numeric_data)

    # 类别距离：汉明距离（相同为0，不同为1），再除以类别数
    cat_dist = np.zeros((n, n), dtype=np.float32)
    for col_idx in range(categorical_data.shape[1]):
        col = categorical_data[:, col_idx]
        col_dist = (col[:, None] != col[None, :]).astype(np.float32)
        cat_dist += col_dist
    cat_dist = cat_dist / categorical_data.shape[1]

    total_dist = num_dist + cat_dist
    # 确保对称
    total_dist = (total_dist + total_dist.T) / 2
    np.fill_diagonal(total_dist, 0)

    elapsed = time.time() - start_time
    print(f"   ✅ Gower 距离计算完成，耗时 {elapsed:.1f} 秒")
    return total_dist


def plot_dendrogram(linkage_matrix, group_name, output_dir):
    """绘制树状图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    dendrogram(
        linkage_matrix,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
        truncate_mode='lastp',
        p=30,
        show_contracted=True,
    )
    ax.set_xlabel('样本索引')
    ax.set_ylabel('距离')
    ax.set_title(f'{group_name} 层次聚类树状图')
    plt.tight_layout()
    plt.savefig(output_dir / f'{group_name}_树状图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存树状图: {output_dir / f'{group_name}_树状图.png'}")


def plot_pca_cluster(X_numeric, labels, group_name, output_dir):
    """PCA 可视化聚类结果"""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_numeric)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=labels, cmap='Set1',
                         s=25, alpha=0.7,
                         edgecolors='black', linewidths=0.3)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    ax.set_title(f'{group_name} 层次聚类 PCA 可视化')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{group_name}_PCA散点图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存 PCA 散点图: {output_dir / f'{group_name}_PCA散点图.png'}")


def process_house_clustering_hierarchical(df_house, city, house_type):
    """层次聚类主流程"""
    group_name = f"{city}_{house_type}"
    print(f"\n{'=' * 50}")
    print(f"🌳 层次聚类: {group_name}")
    print(f"{'=' * 50}")

    # ---- 使用最终特征集 ----
    numeric_existing = [col for col in NUMERIC_COLS if col in df_house.columns]
    categorical_existing = [col for col in CATEGORICAL_COLS if col in df_house.columns]

    print(f"   📊 数值特征: {numeric_existing}")
    print(f"   📊 类别特征: {categorical_existing}")
    print(f"   📊 数据总量: {len(df_house)} 条")

    if len(numeric_existing) < 2 or len(categorical_existing) < 2:
        print(f"⚠️ 特征不足，跳过")
        return None

    # ---- 提取数据 ----
    df_sub = df_house[numeric_existing + categorical_existing].copy()

    if 'house_age' in df_sub.columns:
        df_sub['house_age'] = df_sub['house_age'].clip(0, 50)

    for col in numeric_existing:
        df_sub[col] = df_sub[col].fillna(df_sub[col].median())
    for col in categorical_existing:
        df_sub[col] = df_sub[col].fillna('未知').astype(str)

    # ---- 标准化数值列 ----
    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(df_sub[numeric_existing])
    df_sub[numeric_existing] = numeric_data

    # ---- 计算 Gower 距离 ----
    dist_matrix = compute_gower_distance(df_sub, numeric_existing, categorical_existing)
    compressed_dist = squareform(dist_matrix, checks=False)

    # ---- 层次聚类 ----
    print("   ⏳ 执行层次聚类 (average 链接)...")
    start_time = time.time()
    linkage_matrix = linkage(compressed_dist, method='average')
    elapsed = time.time() - start_time
    print(f"   ✅ 聚类完成，耗时 {elapsed:.1f} 秒")

    # ---- 确定最优 K（各簇占比 ≥ 5%） ----
    print("   ⏳ 确定最优聚类数（平衡约束）...")
    best_k = 2
    best_score = -1
    best_labels = None

    for k in range(2, 9):
        labels = fcluster(linkage_matrix, k, criterion='maxclust')
        unique, counts = np.unique(labels, return_counts=True)
        min_ratio = counts.min() / len(labels)

        print(f"      K={k}: 簇数={len(unique)}, 最小簇占比={min_ratio * 100:.1f}%")

        if min_ratio < 0.05:
            print(f"         ⚠️ 跳过：存在过小簇")
            continue

        try:
            score = silhouette_score(numeric_data, labels)
            print(f"         ✅ 轮廓系数={score:.3f}")
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception as e:
            print(f"         ⚠️ 轮廓系数失败: {e}")

    if best_labels is None:
        print(f"   ⚠️ 未找到满足平衡条件的K值，使用 K=2")
        best_k = 2
        best_labels = fcluster(linkage_matrix, 2, criterion='maxclust')
        best_score = silhouette_score(numeric_data, best_labels)
        unique, counts = np.unique(best_labels, return_counts=True)
        print(f"      K=2: 簇1={counts[0] / len(best_labels) * 100:.1f}%, 簇2={counts[1] / len(best_labels) * 100:.1f}%")

    print(f"   ✅ 最优聚类数: {best_k} (轮廓系数={best_score:.3f})")

    final_labels = best_labels

    # ---- 输出 ----
    out_dir = OUTPUT_DIR / group_name
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_dendrogram(linkage_matrix, group_name, out_dir)
    plot_pca_cluster(numeric_data, final_labels, group_name, out_dir)

    # ---- 簇中心解读 ----
    df_sub['cluster'] = final_labels
    numeric_means = df_sub.groupby('cluster')[numeric_existing].mean().round(3)
    categorical_modes = df_sub.groupby('cluster')[categorical_existing].agg(
        lambda x: x.mode()[0] if not x.mode().empty else '未知'
    )

    interpretation = pd.concat([numeric_means, categorical_modes], axis=1)
    interpretation.index.name = '簇'
    interpretation.to_csv(out_dir / f'{group_name}_簇中心解读.csv', encoding='utf-8-sig')

    # ---- 簇解读 ----
    cluster_sizes = df_sub['cluster'].value_counts().sort_index()
    interpretations = []

    for i in sorted(cluster_sizes.index):
        size = cluster_sizes[i]
        means = numeric_means.loc[i]

        unit_price_val = means.get('unit_price', 0)
        house_age_val = means.get('house_age', 0)
        rent_val = means.get('rent_sale_ratio', 0)
        amenity_val = means.get('total_amenity_score', 0)

        # 类型标签
        if unit_price_val > 0.5 and house_age_val < -0.3:
            cluster_type = '高单价次新品质型'
        elif unit_price_val > 0.5 and house_age_val > 0.3:
            cluster_type = '高单价老旧型（区位溢价）'
        elif unit_price_val < -0.5 and house_age_val < -0.3:
            cluster_type = '低单价次新实惠型'
        elif unit_price_val < -0.5 and house_age_val > 0.3:
            cluster_type = '低单价老旧型（价格洼地）'
        elif unit_price_val > 0.3:
            cluster_type = '中高单价型'
        elif unit_price_val < -0.3:
            cluster_type = '中低单价型'
        else:
            cluster_type = '均价均衡型'

        # 租售比
        if rent_val > 0.5:
            rent_desc = '出租占比高'
        elif rent_val < -0.5:
            rent_desc = '出售占比高'
        else:
            rent_desc = '租售均衡'

        # 配套
        if amenity_val > 0.5:
            amenity_desc = '配套完善'
        elif amenity_val < -0.5:
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
            '主要行政区': categorical_modes.loc[i, 'district'] if 'district' in categorical_modes.columns else '未知',
            '主要物业类型': categorical_modes.loc[i, 'property_type'] if 'property_type' in categorical_modes.columns else '未知',
            '主要装修': categorical_modes.loc[i, 'decoration'] if 'decoration' in categorical_modes.columns else '未知',
            '主要楼层': categorical_modes.loc[i, 'floor_type'] if 'floor_type' in categorical_modes.columns else '未知',
            '主要朝向': categorical_modes.loc[i, 'orientation'] if 'orientation' in categorical_modes.columns else '未知',
            '单价(标准化)': f"{unit_price_val:.2f}",
            '房龄(标准化)': f"{house_age_val:.2f}"
        })

    interp_df = pd.DataFrame(interpretations)
    interp_df.to_csv(out_dir / f'{group_name}_簇解读表.csv', index=False, encoding='utf-8-sig')

    # ---- 保存聚类标签 ----
    df_house['cluster'] = final_labels
    df_house[['house_id', 'city', 'community_id', 'cluster']].to_csv(
        out_dir / f'{group_name}_聚类标签.csv', index=False, encoding='utf-8-sig'
    )

    print(f"✅ 层次聚类完成: {group_name}")
    print(f"   K={best_k}, 样本数={len(df_sub)}, 轮廓系数={best_score:.3f}")

    return {
        'group': group_name,
        'optimal_k': best_k,
        'n_samples': len(df_sub),
        'silhouette': best_score
    }


def main():
    print("=" * 60)
    print("🚀 层次聚类分析 (Gower 距离)")
    print("   数据来源: _complete.csv")
    print("   数值特征: unit_price | house_age | rent_sale_ratio | total_amenity_score")
    print("   类别特征: property_type | decoration | floor_type | district | orientation")
    print("=" * 60)

    groups = [d for d in MODELING_DIR.iterdir() if d.is_dir()]
    if not groups:
        print("❌ 未找到建模数据目录")
        return

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

        result = process_house_clustering_hierarchical(df_house, city, house_type)
        if result:
            summary.append(result)

    print("\n" + "=" * 60)
    print("📊 层次聚类汇总")
    print("=" * 60)
    if summary:
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(OUTPUT_DIR / '层次聚类汇总.csv', index=False, encoding='utf-8-sig')

    print(f"\n✅ 层次聚类完成！结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()