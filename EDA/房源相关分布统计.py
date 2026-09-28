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
OUTPUT_DIR = BASE_DIR / "EDA" / "results" / "分布统计" / "房源"
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
sns.set_style("whitegrid", rc={'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'PingFang SC']})

def load_data():
    """从特征工程输出加载各城市各类型房源独立表并合并"""
    all_house = []
    for city in ["杭州", "金华", "临沂"]:
        for house_type in ["二手房", "租房"]:
            sub_dir = FE_DIR / f"{city}_{house_type}"
            house_file = sub_dir / f"{city}_{house_type}_house.csv"
            if house_file.exists():
                df = pd.read_csv(house_file, encoding='utf-8-sig')
                all_house.append(df)
                print(f"📂 加载 {city} {house_type}: {len(df)} 条")
            else:
                print(f"⚠️ 未找到 {city}_{house_type}_house.csv")
    if not all_house:
        print("❌ 未找到特征工程输出的房源数据")
        return None
    df = pd.concat(all_house, ignore_index=True)
    # 数值列转换
    numeric_cols = ['total_price', 'area', 'build_year', 'house_age', 'unit_price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"📂 总计加载房源数据: {len(df)} 条")
    return df

# ================== 绘图函数 ==================
def plot_district_counts(df):
    for city in df['city'].unique():
        sub = df[df['city'] == city]
        dist_counts = sub.groupby(['district', 'house_type']).size().unstack(fill_value=0)
        dist_counts['合计'] = dist_counts.sum(axis=1)
        dist_counts = dist_counts[dist_counts['合计'] >= 10]
        if dist_counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        dist_counts[['二手房', '租房']].plot(kind='bar', ax=ax, color=['#4CAF50', '#FF9800'])
        ax.set_title(f'{city} 各行政区房源数量', fontsize=18)
        ax.set_xlabel('行政区', fontsize=16)
        ax.set_ylabel('频数', fontsize=16)
        ax.legend(title='房源类型', fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=11)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{city}_行政区房源数量.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存: {city}_行政区房源数量.png")

def plot_price_boxplot(df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, htype in zip(axes, ['二手房', '租房']):
        sub = df[df['house_type'] == htype].dropna(subset=['total_price'])
        if not sub.empty:
            sns.boxplot(x='city', y='total_price', data=sub, ax=ax)
            ax.set_title(f'{htype} 各城市价格分布（箱线图）', fontsize=18)
            ax.set_xlabel('城市', fontsize=16)
            ax.set_ylabel('总价 (万元)' if htype == '二手房' else '月租金 (元/月)', fontsize=16)
            ax.tick_params(axis='both', labelsize=14)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=16)
            ax.set_title(f'{htype}', fontsize=18)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '各城市价格箱线图_分类型.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 保存: 各城市价格箱线图_分类型.png")

def plot_area_boxplot(df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, htype in zip(axes, ['二手房', '租房']):
        sub = df[df['house_type'] == htype].dropna(subset=['area'])
        if not sub.empty:
            sns.boxplot(x='city', y='area', data=sub, ax=ax)
            ax.set_title(f'{htype} 各城市面积分布（箱线图）', fontsize=18)
            ax.set_xlabel('城市', fontsize=16)
            ax.set_ylabel('面积 (㎡)', fontsize=16)
            ax.tick_params(axis='both', labelsize=14)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=16)
            ax.set_title(f'{htype}', fontsize=18)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '各城市面积箱线图_分类型.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 保存: 各城市面积箱线图_分类型.png")

def plot_house_age_boxplot(df):
    if 'house_age' not in df.columns:
        print("⚠️ 数据中无 house_age 列，跳过房龄统计")
        return
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, htype in zip(axes, ['二手房', '租房']):
        sub = df[df['house_type'] == htype].dropna(subset=['house_age'])
        if not sub.empty:
            sns.boxplot(x='city', y='house_age', data=sub, ax=ax)
            ax.set_title(f'{htype} 各城市房龄分布（箱线图）', fontsize=18)
            ax.set_xlabel('城市', fontsize=16)
            ax.set_ylabel('房龄 (年)', fontsize=16)
            ax.tick_params(axis='both', labelsize=14)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=16)
            ax.set_title(f'{htype}', fontsize=18)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '各城市房龄箱线图_分类型.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 保存: 各城市房龄箱线图_分类型.png")

def plot_categorical_percentage(df, col, title):
    if col not in df.columns:
        print(f"⚠️ 列 {col} 不存在，跳过")
        return

    data = []
    for city in df['city'].unique():
        sub = df[df['city'] == city].dropna(subset=[col])
        if sub.empty:
            continue
        total = len(sub)
        freq = sub[col].value_counts()
        for cat, count in freq.items():
            data.append({'城市': city, '分类': cat, '百分比': count / total * 100})

    if not data:
        print(f"⚠️ 列 {col} 无数据，跳过")
        return

    plot_df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x='城市', y='百分比', hue='分类', data=plot_df, ax=ax)
    ax.set_title(f'各城市 {title} 构成对比', fontsize=18)
    ax.set_xlabel('城市', fontsize=16)
    ax.set_ylabel('百分比 (%)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(title=title, fontsize=14, bbox_to_anchor=(1.02, 1), loc='upper left')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'各城市_{col}_百分比对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存: 各城市_{col}_百分比对比.png")

def plot_ownership_type_distribution(df):
    data = []
    for city in df['city'].unique():
        sub = df[df['city'] == city].dropna(subset=['ownership_type'])
        freq = sub['ownership_type'].value_counts()
        for otype, count in freq.items():
            data.append({'城市': city, '产权类型': otype, '房源数量': count})
    if not data:
        print("⚠️ 无产权类型数据，跳过")
        return
    plot_df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x='产权类型', y='房源数量', hue='城市', data=plot_df, ax=ax)
    ax.set_title('各城市房源产权类型分布对比', fontsize=18)
    ax.set_xlabel('产权类型', fontsize=16)
    ax.set_ylabel('房源数量', fontsize=16)
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.legend(title='城市', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '各城市房源产权类型分布对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 保存: 各城市房源产权类型分布对比.png")

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
                '房源数量': count,
                '占比(%)': count / total * 100
            })
    if not data:
        print("⚠️ 无产权年限数据，跳过")
        return
    plot_df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1 = axes[0]
    sns.barplot(x='产权年限', y='房源数量', hue='城市', data=plot_df, ax=ax1)
    ax1.set_title('各城市房源产权年限分布（数量）', fontsize=16)
    ax1.set_xlabel('产权年限', fontsize=14)
    ax1.set_ylabel('房源数量', fontsize=14)
    ax1.tick_params(axis='x', rotation=0)
    for p in ax1.patches:
        if p.get_height() > 0:
            ax1.annotate(f'{int(p.get_height())}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=10)

    ax2 = axes[1]
    sns.barplot(x='产权年限', y='占比(%)', hue='城市', data=plot_df, ax=ax2)
    ax2.set_title('各城市房源产权年限分布（占比）', fontsize=16)
    ax2.set_xlabel('产权年限', fontsize=14)
    ax2.set_ylabel('占比 (%)', fontsize=14)
    ax2.tick_params(axis='x', rotation=0)
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.annotate(f'{height:.1f}%',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '各城市房源产权年限分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 保存: 各城市房源产权年限分布.png")

    pivot_table = plot_df.pivot_table(
        index='城市', columns='产权年限', values='房源数量', fill_value=0
    )
    pivot_table['总计'] = pivot_table.sum(axis=1)
    pivot_table.to_csv(OUTPUT_DIR / '各城市房源产权年限统计.csv', encoding='utf-8-sig')
    print("✅ 保存: 各城市房源产权年限统计.csv")

def main():
    print("🚀 开始房源级别分布统计...")
    df = load_data()
    if df is None:
        return

    plot_district_counts(df)
    plot_price_boxplot(df)
    plot_area_boxplot(df)
    plot_house_age_boxplot(df)

    plot_categorical_percentage(df, 'decoration', '装修程度')
    plot_categorical_percentage(df, 'elevator', '电梯配置')
    plot_categorical_percentage(df, 'floor_type', '楼层类型')
    plot_categorical_percentage(df, 'property_type', '物业类型')
    plot_categorical_percentage(df, 'orientation', '朝向')

    plot_ownership_type_distribution(df)
    plot_property_years_distribution(df)

    print(f"\n✅ 所有房源分析图表已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()