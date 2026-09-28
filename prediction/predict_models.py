# prediction/郭沛祺-predict_global_enhanced.py
# 全局预测模型（增强版）
# 所有多分类变量统一使用 Label Encoding（不做 One-Hot）
# 新特征1: 剩余产权年限 = property_years - house_age
# 新特征2: is_core = 是否核心区
# 模型: 随机森林 + XGBoost + 集成（平均）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
import os

warnings.filterwarnings('ignore')

# ================== 配置 ==================
BASE_DIR = Path(__file__).parent.parent
FE_DIR = BASE_DIR / "feature_engineering" / "modeling_data"
OUTPUT_DIR = Path(__file__).parent / "prediction_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ['杭州', '金华', '临沂']
HOUSE_TYPES = ['二手房', '租房']

# 所有多分类变量统一 Label Encoding
CITY_CODE = {'杭州': 2, '金华': 1, '临沂': 0}
PROPERTY_TYPE_CODE = {'普通住宅': 0, '公寓': 1, '别墅': 2, '商住楼': 3}
DECORATION_CODE = {'毛坯': 0, '简单装修': 1, '精装修': 2, '豪华装修': 3}
FLOOR_TYPE_CODE = {'低层': 0, '中层': 1, '高层': 2, '未知': 3}

RANDOM_STATE = 42
TEST_SIZE = 0.3
SIGMA = 3

# ================== 核心区映射 ==================
CORE_DISTRICTS = {
    '杭州': ['拱墅', '上城', '滨江', '西湖', '钱塘'],
    '金华': ['婺城'],
    '临沂': ['兰山', '北城新区']
}

def is_core_district(city, district):
    if pd.isna(district) or district == '未知' or district == '':
        return 0
    return 1 if district in CORE_DISTRICTS.get(city, []) else 0

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


def load_all_data():
    all_dfs = []
    for city in CITIES:
        for htype in HOUSE_TYPES:
            group_name = f"{city}_{htype}"
            file_path = FE_DIR / group_name / f"{group_name}_complete.csv"
            if not file_path.exists():
                print(f"⚠️ 文件不存在: {file_path}")
                continue
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            df['city'] = city
            df['house_type'] = htype
            all_dfs.append(df)
            print(f"📂 加载 {group_name}: {len(df)} 条")
    if not all_dfs:
        raise FileNotFoundError("未找到任何完整数据文件")
    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"📊 合并后总记录数: {len(df_all)}")
    return df_all


def engineer_features(df):
    df_eng = df.copy()

    # ---- 1. 剩余产权年限 ----
    if 'property_years' in df_eng.columns:
        df_eng['property_years'] = pd.to_numeric(df_eng['property_years'], errors='coerce')
    else:
        df_eng['property_years'] = 70

    if 'house_age' not in df_eng.columns:
        df_eng['house_age'] = 0

    df_eng['remaining_lease'] = df_eng['property_years'] - df_eng['house_age']
    df_eng['remaining_lease'] = df_eng['remaining_lease'].clip(0, 70)

    for city in df_eng['city'].unique():
        for ptype in df_eng['property_type'].unique():
            mask = (df_eng['city'] == city) & (df_eng['property_type'] == ptype) & (df_eng['remaining_lease'].isna())
            if mask.any():
                mode_val = df_eng.loc[(df_eng['city'] == city) & (df_eng['property_type'] == ptype), 'remaining_lease'].mode()
                if not mode_val.empty and pd.notna(mode_val.iloc[0]):
                    df_eng.loc[mask, 'remaining_lease'] = mode_val.iloc[0]

    df_eng['remaining_lease'] = df_eng['remaining_lease'].fillna(70)

    # ---- 2. 是否核心区 ----
    df_eng['is_core'] = df_eng.apply(
        lambda row: is_core_district(row['city'], row['district']),
        axis=1
    )

    # ---- 3. 所有多分类变量 Label Encoding ----
    df_eng['city_code'] = df_eng['city'].map(CITY_CODE).fillna(1).astype(int)
    df_eng['property_type_code'] = df_eng['property_type'].map(PROPERTY_TYPE_CODE).fillna(0).astype(int)
    df_eng['decoration_code'] = df_eng['decoration'].map(DECORATION_CODE).fillna(2).astype(int)
    df_eng['floor_type_code'] = df_eng['floor_type'].map(FLOOR_TYPE_CODE).fillna(1).astype(int)

    print(f"   📊 核心区房源占比: {df_eng['is_core'].mean() * 100:.1f}%")
    print(f"   📊 剩余产权年限均值: {df_eng['remaining_lease'].mean():.1f} 年")

    return df_eng


def remove_outliers(df, target_col):
    mean_val = df[target_col].mean()
    std_val = df[target_col].std()
    if std_val > 0:
        upper_bound = mean_val + SIGMA * std_val
        lower_bound = max(0, mean_val - SIGMA * std_val)
        before = len(df)
        df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
        print(f"   🗑️ 剔除 {before - len(df)} 条异常值 ({SIGMA}σ 截尾)")
    return df


def prepare_features(df, is_rent):
    # ---- 全部数值化，无One-Hot ----
    base_features = [
        'remaining_lease', 'is_core', 'area',
        'city_code', 'property_type_code', 'decoration_code', 'floor_type_code'
    ]

    X = df[base_features].copy()
    if is_rent:
        target = 'total_price'
    else:
        target = 'unit_price'

    y = df[target].copy()

    # 剔除缺失值
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    # 3σ截尾
    df_temp = pd.concat([X, y.to_frame('target')], axis=1)
    df_temp = remove_outliers(df_temp, 'target')
    X = df_temp[base_features]
    y = df_temp['target']

    # 填充缺失值（保障所有特征数值化）
    for col in X.columns:
        if X[col].dtype == 'object':
            mode_val = X[col].mode()[0] if not X[col].mode().empty else 0
            X[col] = X[col].fillna(mode_val)
        else:
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0
            X[col] = X[col].fillna(median_val)

    X = X.astype(float)

    # 因变量取对数
    y_log = np.log(y).astype(float)

    print(f"   📊 特征数: {X.shape[1]}, 有效样本数: {len(y_log)}")
    print(f"      features: {X.columns.tolist()}")
    return X, y_log


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, X_test, y_test):
    print("   🌲 训练随机森林...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print(f"      最佳参数: {grid_search.best_params_}")

    y_pred = best_rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    return best_rf, metrics, importance


def train_xgboost(X_train, y_train, X_test, y_test):
    print("   🚀 训练 XGBoost...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
    }
    xgb = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_
    print(f"      最佳参数: {grid_search.best_params_}")

    y_pred = best_xgb.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_xgb.feature_importances_
    }).sort_values('importance', ascending=False)

    metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    return best_xgb, metrics, importance


def train_ensemble(rf_model, xgb_model, X_test):
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    ensemble_pred = (rf_pred + xgb_pred) / 2
    return ensemble_pred


def evaluate_model(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


def plot_feature_importance(importance_df, title, save_path, top_n=15):
    possible_importance_cols = ['importance', 'Importance', 'avg_importance', 'gain', 'weight']
    importance_col = None
    for col in possible_importance_cols:
        if col in importance_df.columns:
            importance_col = col
            break

    if importance_col is None:
        print(f"   ⚠️ 未找到重要性列，实际列名: {importance_df.columns.tolist()}")
        return

    if 'feature' not in importance_df.columns:
        print(f"   ⚠️ 未找到 'feature' 列")
        return

    importance_df = importance_df.rename(columns={importance_col: 'importance'})
    importance_df = importance_df.sort_values('importance', ascending=False)
    top = importance_df.head(top_n)

    if top.empty:
        print("   ⚠️ 特征重要性数据为空")
        return

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top)))[::-1]
    ax.barh(top['feature'], top['importance'], color=colors)
    ax.set_xlabel('重要性')
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存特征重要性图: {save_path}")


def run_prediction_for_market(df, market_type):
    is_rent = (market_type == '租房')
    print(f"\n{'='*50}")
    print(f"📈 预测: {market_type} 全局增强模型")
    print(f"{'='*50}")

    df_market = df[df['house_type'] == market_type].copy()
    print(f"   原始样本数: {len(df_market)}")

    df_market = engineer_features(df_market)

    X, y = prepare_features(df_market, is_rent)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"   训练集: {len(X_train)}, 测试集: {len(X_test)}")

    rf_model, rf_metrics, rf_importance = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_metrics, xgb_importance = train_xgboost(X_train, y_train, X_test, y_test)

    ensemble_pred = train_ensemble(rf_model, xgb_model, X_test)
    ensemble_metrics = evaluate_model(y_test, ensemble_pred, '集成(RF+XGB)')

    print(f"\n   📊 测试集性能:")
    print(f"      随机森林: RMSE={rf_metrics['RMSE']:.4f}, MAE={rf_metrics['MAE']:.4f}, R²={rf_metrics['R2']:.4f}")
    print(f"      XGBoost:  RMSE={xgb_metrics['RMSE']:.4f}, MAE={xgb_metrics['MAE']:.4f}, R²={xgb_metrics['R2']:.4f}")
    print(f"      集成模型: RMSE={ensemble_metrics['RMSE']:.4f}, MAE={ensemble_metrics['MAE']:.4f}, R²={ensemble_metrics['R2']:.4f}")

    rf_importance.columns = ['feature', 'random_forest']
    xgb_importance.columns = ['feature', 'xgboost']
    combined_importance = pd.merge(rf_importance, xgb_importance, on='feature', how='outer').fillna(0)
    combined_importance['avg_importance'] = (combined_importance['random_forest'] + combined_importance['xgboost']) / 2
    combined_importance = combined_importance.sort_values('avg_importance', ascending=False)

    out_dir = OUTPUT_DIR / market_type
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([
        {'model': '随机森林', **rf_metrics},
        {'model': 'XGBoost', **xgb_metrics},
        ensemble_metrics
    ])
    metrics_df.to_csv(out_dir / f'{market_type}_预测性能.csv', index=False, encoding='utf-8-sig')
    combined_importance.to_csv(out_dir / f'{market_type}_特征重要性.csv', index=False, encoding='utf-8-sig')
    plot_feature_importance(
        combined_importance, f'{market_type} 特征重要性（平均）',
        out_dir / f'{market_type}_特征重要性.png'
    )

    print(f"\n   📌 特征重要性 Top 5:")
    for i, row in combined_importance.head(5).iterrows():
        print(f"      {row['feature']}: {row['avg_importance']:.4f}")

    return {
        'market': market_type,
        'rf_metrics': rf_metrics,
        'xgb_metrics': xgb_metrics,
        'ensemble_metrics': ensemble_metrics,
        'n_samples': len(df_market)
    }


def main():
    print("=" * 60)
    print("🚀 全局预测模型（增强版）")
    print("   新特征1: 剩余产权年限 = property_years - house_age")
    print("   新特征2: is_core = 是否核心区")
    print("   编码策略: 所有多分类变量统一 Label Encoding（不做 One-Hot）")
    print("   模型: 随机森林 + XGBoost + 集成（平均）")
    print("=" * 60)

    df_all = load_all_data()

    results = []
    for market_type in ['二手房', '租房']:
        result = run_prediction_for_market(df_all, market_type)
        results.append(result)

    print("\n" + "=" * 60)
    print("📊 汇总表（测试集）")
    print("=" * 60)
    summary_rows = []
    for r in results:
        summary_rows.append({'市场': r['market'], '模型': '随机森林', **r['rf_metrics']})
        summary_rows.append({'市场': r['market'], '模型': 'XGBoost', **r['xgb_metrics']})
        summary_rows.append({'市场': r['market'], '模型': '集成(RF+XGB)', **r['ensemble_metrics']})
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / '全局预测汇总.csv', index=False, encoding='utf-8-sig')

    print(f"\n✅ 预测完成！结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()