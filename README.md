# 项目代码结构与运行说明

本文档说明项目的目录结构及爬虫模块的使用方法，后续数据清洗、特征工程、建模等步骤请参考各子目录下的详细说明。

## 一、项目根目录结构
项目根目录/
├── data/ # 所有数据（原始、中间、最终）
├── spider/ # 爬虫脚本
├── data_clean/ # 数据清洗脚本
├── data_after_clean/ # 清洗后的数据
├── feature_engineering/ # 特征工程脚本
├── EDA/ # 探索性数据分析
├── cluster/ # 聚类分析
├── regression/ # 回归建模
└── prediction/ # 预测与结果输出

---

## 二、爬虫模块（spider）

### 2.1 城市脚本说明

每个城市文件夹（如 `杭州/`）内包含 **5 个按顺序执行的脚本**，用于采集小区信息、房源链接及房源详情。脚本命名规则为：

- `{城市拼音}_community_links.py`   – 获取小区列表及详情页URL
- `{城市拼音}_community_details.py` – 爬取小区详细信息（含周边配套）
- `{城市拼音}_house_links.py`       – 爬取二手房/租房房源链接
- `{城市拼音}_sale_details.py`      – 爬取二手房房源详细信息
- `{城市拼音}_rent_details.py`       – 爬取租房房源详细信息

### 2.2 运行顺序与数据流向

| 步骤 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 1 | `community_links` | 无（直接从安居客列表页爬取） | `data/{城市}/community/{城市}小区_详情链接.csv` |
| 2 | `community_details` | `{城市}小区_详情链接.csv` | `data/{城市}/community/{城市}小区信息.csv` |
| 3 | `house_links` | `{城市}小区_详情链接.csv` 及 `{城市}小区信息.csv` | `data/{城市}/sale/{城市}二手房_详情链接.csv`<br>`data/{城市}/rent/{城市}租房_详情链接.csv` |
| 4 | `sale_details` | `{城市}二手房_详情链接.csv` | `data/{城市}/sale/{城市}二手房_房源信息.csv` |
| 5 | `rent_details` | `{城市}租房_详情链接.csv` | `data/{城市}/rent/{城市}租房_房源信息.csv` |

### 2.3 运行示例（以杭州为例）

假设当前目录为项目根目录，依次执行以下命令：

```bash

# 1. 爬取小区链接
python spider/杭州/hangzhou_community_links.py

# 2. 爬取小区详细信息
python spider/杭州/hangzhou_community_details.py

# 3. 爬取房源链接（二手房 + 租房）
python spider/杭州/hangzhou_house_links.py

# 4. 爬取二手房详情
python spider/杭州/hangzhou_sale_details.py

# 5. 爬取租房详情
python spider/杭州/hangzhou_rent_details.py

```

### 2.4 行政区补爬修复
由于部分小区详情页的行政区（district）字段可能未成功提取，在完成所有城市的小区详情爬取后，运行根目录下的 district_fix.py 脚本进行补爬。该脚本会遍历配置的城市列表，重新访问每个小区的详情页并更新 {城市}小区_详情链接.csv 中的 district 列。

```bash

# 从项目根目录运行
python spider/district_fix.py

```
### 三、数据清洗模块（data_clean）

data_clean/ 目录下包含 7 个按顺序执行的脚本，用于对爬虫采集的原始数据进行清洗、填充、标准化和提取，最终生成可用于建模分析的完整数据集。

### 3.1 脚本列表

| 脚本 | 步骤 | 功能描述 |
|------|------|----------|
| step1_clean.py | 1 | 去重与文本清洗（去除换行符、空格） |
| step2_fill.py | 2 | 缺失值填充（行政区、建筑年份、电梯信息两轮填充） |
| step3_property.py | 3 | 物业类型标准化（房源级 + 小区级覆盖） |
| step4_ownership.py | 4 | 产权类型与产权年限清洗（小区 → 房源映射） |
| step5_floor.py | 5 | 楼层数据提取（解析"类型/总层数"） |
| step6_orientation.py | 6 | 朝向清洗（统一为东、南、西、北） |
| step7_extract.py | 7 | 完整数据提取（房源严检 + 小区严检，生成最终数据集） |

### 3.2 输入数据来源

所有脚本的输入均来自爬虫模块输出的原始文件，位于 data/{城市}/{category}/ 下：
小区数据：data/{城市}/community/{城市}小区信息.csv
二手房数据：data/{城市}/sale/{城市}二手房_房源信息.csv
租房数据：data/{城市}/rent/{城市}租房_房源信息.csv

### 3.3 中间文件与输出目录

所有清洗后的中间文件和最终结果均输出至 data_after_clean/ 目录，保持与 data/ 相同的子目录结构：
data_after_clean/
├── {城市}/
│   ├── community/
│   │   └── {城市}小区信息_cleaned.csv          # 步骤1输出
│   ├── sale/
│   │   ├── {城市}二手房_房源信息_cleaned.csv   # 步骤1输出
│   │   ├── {城市}二手房_房源信息_filled.csv    # 步骤2输出
│   │   ├── {城市}二手房_房源信息_filled_property.csv       # 步骤3输出
│   │   ├── {城市}二手房_房源信息_filled_property_ownership.csv  # 步骤4输出
│   │   ├── {城市}二手房_房源信息_final_floor.csv            # 步骤5输出
│   │   └── {城市}二手房_房源信息_final_floor_ori.csv        # 步骤6输出
│   └── rent/
│       └── ...（同上结构）
├── 完整小区数据.csv          # 步骤7最终输出
└── 完整房源数据.csv          # 步骤7最终输出

### 3.4 执行顺序与示例

| 步骤 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 1 | step1_clean.py | data/{城市}/.../*.csv | data_after_clean/{城市}/.../*_cleaned.csv |
| 2 | step2_fill.py | *_cleaned.csv + {城市}小区_详情链接.csv（用于 district） | *_filled.csv |
| 3 | step3_property.py | *_filled.csv + 小区清洗表 | *_filled_property.csv |
| 4 | step4_ownership.py | *_filled_property.csv + 小区清洗表 | *_filled_property_ownership.csv |
| 5 | step5_floor.py | *_filled_property_ownership.csv | *_final_floor.csv |
| 6 | step6_orientation.py | *_final_floor.csv | *_final_floor_ori.csv |
| 7 | step7_extract.py | 所有城市的 *_final_floor_ori.csv 及小区表 | 完整房源数据.csv、完整小区数据.csv |

```bash

# 步骤1：去重与文本清洗
python data_clean/step1_clean.py

# 步骤2：缺失值填充
python data_clean/step2_fill.py

# 步骤3：物业类型标准化
python data_clean/step3_property.py

# 步骤4：产权类型与产权年限清洗
python data_clean/step4_ownership.py

# 步骤5：楼层数据提取
python data_clean/step5_floor.py

# 步骤6：朝向清洗
python data_clean/step6_orientation.py

# 步骤7：完整数据提取
python data_clean/step7_extract.py

```
### 四、特征工程模块（feature_engineering）

feature_engineering/ 目录下包含 1 个核心脚本，用于构造派生特征、剔除冗余字段，并输出供 EDA 和建模使用的基础数据表。

### 4.1 脚本说明

- `feature_engineering.py`	- 加载清洗后的完整数据，构造派生特征（房龄、单价、租售比、配套得分等），输出按城市-房源类型分组的三个数据表

### 4.2 输入数据

| 数据文件 | 路径 |
|----------|------|
| 完整房源数据 | data_after_clean/完整房源数据.csv |
| 完整小区数据 | data_after_clean/完整小区数据.csv |

### 4.3 输出目录与文件

所有输出文件保存在 feature_engineering/modeling_data/ 下，按 {城市}_{房源类型} 分组（如 杭州_二手房、杭州_租房），每组输出 3 个表：

feature_engineering/modeling_data/
├── 杭州_二手房/
│   ├── 杭州_二手房_complete.csv    # 完整合并表（房源+小区所有字段）
│   ├── 杭州_二手房_house.csv       # 房源独立表
│   └── 杭州_二手房_community.csv   # 小区独立表
├── 杭州_租房/
│   └── ...（同上）
├── 金华_二手房/
│   └── ...
└── 特征工程汇总.csv               # 各组样本数汇总

| 派生特征 | 计算方式 |
|----------|----------|
| `house_age` | 房龄 = 2026 - build_year |
| `unit_price` | 单价 = total_price / area |
| `rent_sale_ratio` | 租售比 = nrp / nsh |
| `score_metro` | 地铁配套得分 = log1p(around_地铁_count) × 3 |
| `score_school` | 学校配套得分 = log1p(around_学校_count) × 2.5 |
| `score_hospital` | 医院配套得分 = log1p(around_医院_count) × 2.0 |
| `score_shopping` | 购物配套得分 = log1p(around_购物_count) × 1.6 |
| `score_dining` | 餐饮配套得分 = log1p(around_餐饮_count) × 1.3 |
| `total_amenity_score` | 五项配套得分之和 |

# 在项目根目录执行
```bash

python feature_engineering/feature_engineering.py

```
### 五、探索性数据分析模块（EDA）

EDA/ 目录下包含 3 个分析脚本，用于对特征工程输出的数据进行分布统计、相关性分析和可视化，输出图表及统计结果。

### 5.1 脚本列表

| 脚本 | 功能 |
|------|------|
| `房源相关分布统计.py` | 房源维度分布统计（行政区、价格、面积、房龄、装修、朝向、产权等） |
| `小区相关分布统计.py` | 小区维度分布统计（均价、在售/在租、物业类型、配套设施覆盖率、产权等） |
| `相关性分析.py` | 全国及分城市数值变量相关性热力图、分类变量ANOVA、城市间差异检验 |

### 5.2 输入数据

| 数据文件 | 路径 |
|----------|------|
| 各组完整合并表 | `feature_engineering/modeling_data/{城市}_{类型}_complete.csv` |
| 各组房源独立表 | `feature_engineering/modeling_data/{城市}_{类型}_house.csv` |
| 各组小区独立表 | `feature_engineering/modeling_data/{城市}_{类型}_community.csv` |

### 5.3 输出目录与文件

所有输出文件保存在 EDA/results/ 下：
EDA/results/
├── 分布统计/
│   ├── 房源/
│   │   ├── {城市}_行政区房源数量.png
│   │   ├── 各城市价格箱线图_分类型.png
│   │   ├── 各城市面积箱线图_分类型.png
│   │   ├── 各城市房龄箱线图_分类型.png
│   │   ├── 各城市_{分类变量}_百分比对比.png
│   │   ├── 各城市房源产权类型分布对比.png
│   │   └── 各城市房源产权年限分布.png
│   └── 社区/
│       ├── 各城市小区均价分布.png
│       ├── 各城市在售_在租平均数量.png
│       ├── 各城市物业类型分布对比.png
│       ├── 各城市配套设施覆盖率对比.png
│       ├── 各城市产权类型分布对比.png
│       └── 各城市产权年限分布.png
└── 相关性/
    ├── 全国_{类型}_相关性热力图.png
    ├── 全国_{类型}_相关系数矩阵.csv
    ├── 全国_{类型}_类别变量ANOVA.csv
    ├── {城市}/
    │   ├── {城市}_{类型}_相关性热力图.png
    │   ├── {城市}_{类型}_相关系数矩阵.csv
    │   └── {城市}_{类型}_类别变量ANOVA.csv
    ├── 所有分析_类别变量ANOVA汇总.csv
    └── 城市间差异/
        ├── 二手房/
        │   ├── 二手房_连续变量_城市间ANOVA差异.csv
        │   └── 二手房_连续变量_Tukey事后检验详细.csv
        └── 租房/
            ├── 租房_连续变量_城市间ANOVA差异.csv
            └── 租房_连续变量_Tukey事后检验详细.csv

### 5.4 各脚本输出说明

| 脚本 | 输出内容 |
|------|----------|
| `房源相关分布统计.py` | 各城市行政区房源数量条形图、价格/面积/房龄箱线图（分类型）、装修/电梯/楼层/物业/朝向/产权类型/产权年限百分比对比图 |
| `小区相关分布统计.py` | 各城市小区均价分布直方图、在售/在租平均数量对比、物业类型分布、配套设施覆盖率、产权类型分布、产权年限分布 |
| `相关性分析.py` | 全国及分城市数值变量相关性热力图（含相关系数矩阵CSV）、分类变量对价格的ANOVA分析（含效应量）、城市间连续变量ANOVA+Tukey HSD事后检验、分类变量卡方检验 |

### 5.5 运行命令

```bash

# 房源分布统计
python EDA/房源相关分布统计.py

# 小区分布统计
python EDA/小区相关分布统计.py

# 相关性分析
python EDA/相关性分析.py

```

### 六、聚类分析模块（cluster）

cluster/ 目录下包含 3 个脚本，用于聚类前的特征诊断以及两种聚类算法的建模分析。

### 6.1 脚本列表

| 脚本 | 功能 |
|------|------|
| `cluster_feature_diagnostic.py` | 聚类前特征诊断（数值相关性、类别关联性、类别-数值区分度） |
| `modelling_cluster_hierarchical.py` | 层次聚类（Gower 距离 + average linkage） |
| `modelling_cluster_kproto.py` | K-Prototypes 聚类（同时处理数值与类别特征） |

### 6.2 输入数据

| 数据文件 | 路径 |
|----------|------|
| 各组完整合并表 | `feature_engineering/modeling_data/{城市}_{类型}_complete.csv` |


### 6.3 输出目录与文件

cluster/
├── feature_diagnostic_results/
│   ├── {城市}_{类型}/
│   │   ├── 数值相关系数矩阵.csv
│   │   ├── CramersV关联矩阵.csv
│   │   └── ANOVA_p值表.csv
│   └── 特征诊断汇总.csv
├── hierarchical_results/
│   ├── {城市}_{类型}/
│   │   ├── {城市}_{类型}_树状图.png
│   │   ├── {城市}_{类型}_PCA散点图.png
│   │   ├── {城市}_{类型}_簇中心解读.csv
│   │   ├── {城市}_{类型}_簇解读表.csv
│   │   └── {城市}_{类型}_聚类标签.csv
│   └── 层次聚类汇总.csv
└── kproto_results/
    ├── {城市}_{类型}/
    │   ├── {城市}_{类型}_K值选择.png
    │   ├── {城市}_{类型}_PCA散点图.png
    │   ├── {城市}_{类型}_簇解读表.csv
    │   └── {城市}_{类型}_聚类标签.csv
    └── KPrototypes聚类汇总.csv

### 6.4 各脚本说明

| 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `cluster_feature_diagnostic.py` | `_complete.csv` | 数值相关系数矩阵、Cramér's V 关联矩阵、ANOVA p 值表 | 检验 4 个数值特征 + 6 个类别特征，诊断后确定最终特征集 |
| `modelling_cluster_hierarchical.py` | `_complete.csv` | 树状图、PCA散点图、簇中心解读、簇解读表、聚类标签 | 使用 Gower 距离 + average linkage，自动确定最优 K |
| `modelling_cluster_kproto.py` | `_complete.csv` | K值选择图（肘部+轮廓系数）、PCA散点图、簇解读表、聚类标签 | K-Prototypes 算法，同时处理数值与类别特征 |

### 6.5 最终特征集

诊断完成后确定的聚类特征集：

| 类型 | 特征 |
|------|------|
| 数值特征（4个） | `unit_price`、`house_age`、`rent_sale_ratio`、`total_amenity_score` |
| 类别特征（5个） | `property_type`、`decoration`、`floor_type`、`district`、`orientation` |

### 6.6 运行命令

```bash

# 步骤1：聚类前特征诊断
python cluster/cluster_feature_diagnostic.py

# 步骤2：层次聚类
python cluster/modelling_cluster_hierarchical.py

# 步骤3：K-Prototypes 聚类
python cluster/modelling_cluster_kproto.py

```
### 七、回归建模模块（regression）

regression/ 目录下包含 3 个脚本，用于回归建模的数据准备、诊断和模型训练。

### 7.1 脚本列表

| 脚本 | 功能 |
|------|------|
| `regression_prepare.py` | 回归建模数据准备（One-Hot 编码 + 训练/测试集划分） |
| `regression_diagnostic.py` | 回归诊断（VIF 多重共线性、残差检验、低频率虚拟变量检查） |
| `regression_models.py` | 分城市分类型回归建模（根据诊断结果定制特征处理） |

### 7.2 输入数据

| 数据文件 | 路径 |
|----------|------|
| 各组完整合并表 | `feature_engineering/modeling_data/{城市}_{类型}_complete.csv` |

### 7.3 输出目录与文件

regression/
├── modeling_data/ # 数据准备阶段输出
│ ├── {城市}{类型}/
│ │ ├── {城市}{类型}X_train.csv
│ │ ├── {城市}{类型}X_test.csv
│ │ ├── {城市}{类型}y_train.csv
│ │ ├── {城市}{类型}y_test.csv
│ │ ├── {城市}{类型}metadata_train.csv
│ │ ├── {城市}{类型}metadata_test.csv
│ │ └── {城市}{类型}feature_cols.csv
│ └── 回归建模数据准备汇总.csv
├── regression_diagnostic_results/ # 诊断阶段输出
│ ├── {城市}{类型}/
│ │ ├── {城市}_{类型}VIF.csv
│ │ ├── {城市}{类型}低频率虚拟变量.csv
│ │ └── {城市}{类型}诊断结果.csv
│ └── 回归诊断汇总.csv
└── regression_results/ # 建模阶段输出
├── {城市}{类型}full_summary.txt
├── {城市}{类型}coefficients.csv
├── {城市}{类型}_summary.csv
└── 所有模型汇总.csv

### 7.4 各脚本说明

| 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `regression_prepare.py` | `_complete.csv` | X_train/test、y_train/test、metadata、特征列表 | One-Hot 编码（drop_first=True），二值变量直接保留，二手房因变量为 unit_price（排除 area），租房因变量为 total_price（保留 area），test_size=0.3 |
| `regression_diagnostic.py` | X_train、y_train | VIF 表、低频率虚拟变量表、残差诊断结果 | VIF > 10 建议剔除，占比 < 1% 的虚拟变量建议检查/合并，只做诊断不训练最终模型 |
| `regression_models.py` | X_train/test、y_train/test | 模型摘要、系数表、拟合优度汇总 | 根据各组合诊断结果定制特征处理，通用剔除 plot_ratio、greening_rate、elevator_encoded、orientation_*，租房装修类别合并 |

### 7.5 运行命令

```bash

# 步骤1：回归建模数据准备
python regression/regression_prepare.py

# 步骤2：回归诊断（根据结果调整变量）
python regression/regression_diagnostic.py

# 步骤3：回归建模
python regression/regression_models.py

```

### 八、预测与结果输出模块（prediction）
prediction/ 目录下包含 1 个核心脚本，用于训练全局预测模型并输出预测结果。

### 8.1 脚本列表

| 脚本 | 功能 |
|------|------|
| `predict_models.py` | 全局预测模型（增强版），包含特征工程、模型训练与集成预测 |

### 8.2 输入数据

| 数据文件 | 路径 |
|----------|------|
| 各组完整合并表 | `feature_engineering/modeling_data/{城市}_{类型}_complete.csv` |

### 8.3 输出目录与文件

prediction/prediction_results/
├── 二手房/
│ ├── 二手房_预测性能.csv
│ ├── 二手房_特征重要性.csv
│ └── 二手房_特征重要性.png
├── 租房/
│ ├── 租房_预测性能.csv
│ ├── 租房_特征重要性.csv
│ └── 租房_特征重要性.png
└── 全局预测汇总.csv

### 8.4 各脚本说明

| 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `predict_models.py` | 所有 `_complete.csv` | 预测性能指标、特征重要性图、汇总表 | 新增特征：剩余产权年限、是否核心区；Label Encoding（不做 One-Hot）；模型：随机森林 + XGBoost + 集成（平均） |

### 8.5 新增特征说明

| 新增特征 | 计算方式 |
|----------|----------|
| `remaining_lease` | 剩余产权年限 = property_years - house_age |
| `is_core` | 是否核心区（杭州：拱墅/上城/滨江/西湖/钱塘；金华：婺城；临沂：兰山/北城新区） |

### 8.6 编码策略

| 原始变量 | 编码方式 |
|----------|----------|
| city | Label Encoding（杭州=2，金华=1，临沂=0） |
| property_type | Label Encoding（普通住宅=0，公寓=1，别墅=2，商住楼=3） |
| decoration | Label Encoding（毛坯=0，简单装修=1，精装修=2，豪华装修=3） |
| floor_type | Label Encoding（低层=0，中层=1，高层=2，未知=3） |

### 8.7 运行命令

```bash

# 运行全局预测模型
python prediction/predict_models.py

```

### 九、运行环境配置

本项目所有脚本均基于本人电脑上的 Anaconda 虚拟环境运行，Python 解释器路径为：
D:\Anaconda\envs\PythonProject\python.exe