import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from pathlib import Path

# ================== 配置 ==================
BASE_DIR = Path(__file__).parent.parent
FE_DIR = BASE_DIR / "feature_engineering" / "modeling_data"
OUTPUT_DIR = BASE_DIR / "EDA" / "results" / "分布统计" / "社区"
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
sns.set_style("whitegrid", rc={'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'PingFang SC']})

def load_data():
    """从特征工程输出加载各城市的小区独立表并合并"""
    all_comm = []
    for city in ["杭州", "金华", "临沂"]:
        # 小区表不区分房源类型，任选一种即可（优先二手房）
        for house_type in ["二手房", "租房"]:
            sub_dir = FE_DIR / f"{city}_{house_type}"
            comm_file = sub_dir / f"{city}_{house_type}_community.csv"
            if comm_file.exists():
                df = pd.read_csv(comm_file, encoding='utf-8-sig')
                all_comm.append(df)
                break  # 找到一个即可
    if not all_comm:
        print("❌ 未找到特征工程输出的社区数据")
        return None
    df = pd.concat(all_comm, ignore_index=True)
    # 按小区ID去重
    if 'community_id' in df.columns:
        df = df.drop_duplicates(subset=['community_id'], keep='first')
    # 数值列转换（确保类型）
    numeric_cols = ['price', 'build_year', 'nsh', 'nrp', 'property_years',
                    'rent_sale_ratio', 'total_amenity_score']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"📂 加载小区数据: {len(df)} 条")
    return df

def convert_around_to_binary(df):
    around_cols = [col for col in df.columns if col.startswith('around_') and col.endswith('_count')]
    for col in around_cols:
        new_col = col.replace('_count', '_has')
        df[new_col] = (df[col] > 0).astype(int)
    return df, [col.replace('_count', '_has') for col in around_cols]

# ================== 绘图函数 ==================
def plot_price_distribution(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False)
    for ax, city in enumerate(df['city'].unique()):
        sub = df[df['city'] == city].dropna(subset=['price'])
        if not sub.empty:
            sub['price'].hist(bins=30, ax=axes[ax], color='steelblue', edgecolor='black')
            axes[ax].set_title(f'{city} 小区挂牌均价分布')
            axes[ax].set_xlabel('均价 (元/㎡)')
            axes[ax].set_ylabel('频数')
        else:
            axes[ax].text(0.5, 0.5, '无数据', ha='center', va='center')
            axes[ax].set_title(f'{city}')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '各城市小区均价分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 保存: 各城市小区均价分布.png")

def plot_nsh_nrp_distribution(df):
    data = []
    for city in df['city'].unique():
        sub = df[df['city'] == city]
        nsh_mean = sub['nsh'].mean()
        nrp_mean = sub['nrp'].mean()
        data.append({'城市': city, '类型': '在售', '平均数量': nsh_mean})
        data.append({'城市': city, '类型': '在租', '平均数量': nrp_mean})
    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='城市', y='平均数量', hue='类型', data=plot_df, ax=ax, palette=['#4CAF50', '#FF9800'])
    ax.set_title('各城市小区平均在售/在租房源数量', fontsize=14)
    ax.set_xlabel('城市', fontsize=12)
    ax.set_ylabel('平均数量', fontsize=12)
    ax.legend(title='类型')
    ax.tick_params(axis='x', rotation=0)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '各城市在售_在租平均数量.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 保存: 各城市在售_在租平均数量.png")

def plot_property_type(df):
    data = []
    for city in df['city'].unique():
        sub = df[df['city'] == city].dropna(subset=['property_type'])
        freq = sub['property_type'].value_counts()
        for prop_type, count in freq.items():
            data.append({'城市': city, '物业类型': prop_type, '小区数量': count})
    if not data:
        print("⚠️ 无物业类型数据，跳过")
        return
    plot_df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='物业类型', y='小区数量', hue='城市', data=plot_df, ax=ax)
    ax.set_title('各城市物业类型分布对比', fontsize=14)
    ax.set_xlabel('物业类型', fontsize=12)
    ax.set_ylabel('小区数量', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.legend(title='城市')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '各城市物业类型分布对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 保存: 各城市物业类型分布对比.png")

def plot_amenities_comparison(df, has_cols):
    data = []
    for city in df['city'].unique():
        sub = df[df['city'] == city]
        for col in has_cols:
            if col in sub.columns:
                label = col.replace('around_', '').replace('_has', '')
                pct = sub[col].mean() * 100
                data.append({'城市': city, '设施': label, '覆盖率 (%)': pct})
    if not data:
        return
    plot_df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='设施', y='覆盖率 (%)', hue='城市', data=plot_df, ax=ax)
    ax.set_title('各城市周边配套设施覆盖率对比', fontsize=14)
    ax.set_xlabel('设施类型', fontsize=12)
    ax.set_ylabel('覆盖率 (%)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.legend(title='城市')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '各城市配套设施覆盖率对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 保存: 各城市配套设施覆盖率对比.png")

def plot_ownership_type_distribution(df):
    data = []
    for city in df['city'].unique():
        sub = df[df['city'] == city].dropna(subset=['ownership_type'])
        freq = sub['ownership_type'].value_counts()
        for otype, count in freq.items():
            data.append({'城市': city, '产权类型': otype, '小区数量': count})
    if not data:
        print("⚠️ 无产权类型数据，跳过")
        return
    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x='产权类型', y='小区数量', hue='城市', data=plot_df, ax=ax)
    ax.set_title('各城市产权类型分布对比', fontsize=14)
    ax.set_xlabel('产权类型', fontsize=12)
    ax.set_ylabel('小区数量', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.legend(title='城市')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '各城市产权类型分布对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 保存: 各城市产权类型分布对比.png")

def plot_property_years_distribution(df):
    if 'property_years' not in df.columns:
        print("⚠️ 数据中无 property_years 列，跳过产权年限统计")
        return

    data = []
    for city in df['city'].unique():
        sub = df[df['city'] == city].dropna(subset=['property_years'])
        if sub.empty:
            continue
        freq = sub['property_years'].value_counts().sort_index()
        total = len(sub)
        for years, count in freq.items():
            data.append({
                '城市': city,
                '产权年限': f'{int(years)}年',
                '小区数量': count,
                '占比(%)': count / total * 100
            })

    if not data:
        print("⚠️ 无产权年限数据，跳过")
        return

    plot_df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    sns.barplot(x='产权年限', y='小区数量', hue='城市', data=plot_df, ax=ax1)
    ax1.set_title('各城市产权年限分布（数量）', fontsize=14)
    ax1.set_xlabel('产权年限', fontsize=12)
    ax1.set_ylabel('小区数量', fontsize=12)
    ax1.tick_params(axis='x', rotation=0)
    for p in ax1.patches:
        if p.get_height() > 0:
            ax1.annotate(f'{int(p.get_height())}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9)

    ax2 = axes[1]
    sns.barplot(x='产权年限', y='占比(%)', hue='城市', data=plot_df, ax=ax2)
    ax2.set_title('各城市产权年限分布（占比）', fontsize=14)
    ax2.set_xlabel('产权年限', fontsize=12)
    ax2.set_ylabel('占比 (%)', fontsize=12)
    ax2.tick_params(axis='x', rotation=0)
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.annotate(f'{height:.1f}%',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '各城市产权年限分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 保存: 各城市产权年限分布.png")

    pivot_table = plot_df.pivot_table(
        index='城市', columns='产权年限', values='小区数量', fill_value=0
    )
    pivot_table['总计'] = pivot_table.sum(axis=1)
    pivot_table.to_csv(OUTPUT_DIR / '各城市产权年限统计.csv', encoding='utf-8-sig')
    print("✅ 保存: 各城市产权年限统计.csv")

def main():
    print("🚀 开始小区级别探索性分析...")
    df = load_data()
    if df is None:
        return

    df, has_cols = convert_around_to_binary(df)
    print(f"📌 转换了 {len(has_cols)} 个配套设施字段为0-1变量")

    plot_price_distribution(df)
    plot_nsh_nrp_distribution(df)
    plot_property_type(df)
    plot_amenities_comparison(df, has_cols)
    plot_ownership_type_distribution(df)
    plot_property_years_distribution(df)

    print(f"\n✅ 所有小区分析图表已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()