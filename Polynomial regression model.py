import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PolynomialFeatures, MinMaxScaler,  PowerTransformer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import boxcox
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# 全面忽略所有警告
warnings.simplefilter("ignore")
# 加載數據
dt = pd.read_excel("D:/desktop/中興大學/實驗室/機器學習/完整 (發paper)/NiCoV/錢德魯 資料集(35新).xlsx")
print(dt.head())
print("缺失值統計：\n", dt.isnull().sum())

# 數據分佈可視化
sns.set(rc={'figure.figsize': (10, 10)})
sns.displot(dt['Overpotential'], kde=True)
plt.title("Overpotential")
plt.show()

# 特徵與目標變量的關係散點圖
features = ['Temperature', 'Electrolyte', 'Ni', 'Co', 'V', 'Urea']
target = dt['Overpotential']

plt.figure(figsize=(15, 5))
for i, col in enumerate(features):
    plt.subplot(1, len(features), i + 1)
    plt.scatter(dt[col], target, marker='o', alpha=0.7)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Overpotential')
plt.tight_layout()
plt.show()


correlation_matrix = dt.corr().round(2)
# annot = True 讓我們可以把數字標進每個格子裡
sns.heatmap(data=correlation_matrix, annot = True)


# 特徵和目標分離
X = dt.drop('Overpotential', axis=1)
Y = dt['Overpotential']

#計算特徵的重要性
feature_importances = []
for _ in range(100):  # 執行 100 次
    model = GradientBoostingRegressor(random_state=None)
    model.fit(X, Y)
    feature_importances.append(model.feature_importances_)

mean_importances = np.mean(feature_importances, axis=0)
print("平均特徵重要性:", mean_importances)

#繪製雷達圖
# 特徵名稱和重要性
features = ['Temperature', 'Electrolyte', 'Ni', 'Co', 'V', 'Urea']
importance = [0.15790057, 0.74408924, 0.01572956, 0.01586696, 0.01499805, 0.05141562]

# 計算角度
num_features = len(features)
angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
angles += angles[:1]  # 閉合角度
importance += importance[:1]  # 閉合重要性數據

# 繪圖
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# 設置底部圓形背景顏色為淺灰色
ax.set_facecolor('#F0F0F0')  # 淺灰色背景

# 使用不同顏色繪製完整的扇形區域
colors = ['#FF6347', '#FFD700', '#90EE90', '#87CEEB', '#9370DB', '#FFA07A']
for i in range(len(features)):
    # 每個扇形的角度範圍（中心對應標籤）
    theta = np.linspace(angles[i] - np.pi / num_features, angles[i] + np.pi / num_features, 100)
    r = np.array([importance[i]] * len(theta))  # 半徑固定為該特徵的重要性
    ax.fill(np.concatenate([[angles[i]], theta, [angles[i]]]),
            np.concatenate([[0], r, [0]]),  # 圓心對應的半徑為 0
            color=colors[i], alpha=1)  # 設置為完全不透明

# 添加刻度線
ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])  # 增加刻度線
ax.set_yticklabels(['10%', '30%', '50%', '70%', '90%'], color='grey', fontsize=16)

# 添加特徵標籤
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=18, fontweight='bold')

# 美化圖表背景
ax.spines['polar'].set_visible(False)
ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

# 添加圖例，設置圖例框背景顏色為與底部圓相同
legend_labels = [f'{features[i]}: {importance[i] * 100:.1f}%' for i in range(len(features))]
ax.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.3, 1.2), fontsize=14, title="Features", 
          title_fontsize=16, facecolor='#F0F0F0', edgecolor='black')  # 圖例框背景設為淺灰色

# 添加標題
plt.title('Feature Importance Radar Chart', size=24, y=1.1, fontweight='bold')

plt.tight_layout()
plt.show()

# 特徵變換
log_transformer = FunctionTransformer(np.log1p)
sqrt_transformer = FunctionTransformer(np.sqrt)
reciprocal_transformer = FunctionTransformer(np.reciprocal)  # 倒數變換
exponential_transformer = FunctionTransformer(np.exp)  # 指數變換


X['LOG_Temperature'] = log_transformer.fit_transform(X[['Temperature']].values)
X['SQRT_Urea'] = sqrt_transformer.fit_transform(X[['Urea']].values)

X.drop(['Temperature','Urea'], axis=1, inplace=True)



# 修改 ArrayToDataFrameTransformer 类
class ArrayToDataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        """
        初始化转化器并接受列名。
        :param columns: 列名列表，与原始数据一致。
        """
        self.columns = columns

    def fit(self, X, y=None):
        """不执行任何操作，只返回 self。"""
        return self

    def transform(self, X):
        """
        将 numpy.ndarray 转换为 pandas.DataFrame。
        :param X: numpy.ndarray
        :return: pandas.DataFrame
        """
        return pd.DataFrame(X, columns=self.columns)

    def get_feature_names_out(self, input_features=None):
        """返回列名"""
        return self.columns


selector = SelectFromModel(RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42), threshold=0.02)
pipeline = Pipeline([
    ('array_to_df', ArrayToDataFrameTransformer(columns=X.columns)),  # 确保数据框架一致
    ('yeo_johnson', PowerTransformer(method='yeo-johnson')),  # 数据分布归一化（使接近正态分布）
    ('poly', PolynomialFeatures(include_bias=False)),  # 生成多项式特征
    ('scaler', MinMaxScaler()),  # 标准化特征
    ('feature_selection', selector),  # 特征选择
    ('lr', LinearRegression())  # 线性回归模型
])





# 定義參數範圍
param_grid = {
    'poly__degree': [2, 3],  # 多項式次數
    'feature_selection__threshold': ['mean', 'median', 0.01, 0.02, 0.03],  # 特徵重要性閾值
    'feature_selection__estimator__n_estimators': [100, 200, 300],  # 隨機森林的樹數量
    'feature_selection__estimator__max_depth': [None, 5, 10],
}

from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

# 使用 GridSearchCV 搜索最佳參數
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=loo,
    n_jobs=-1
)

# 執行搜索
grid_search.fit(X, Y)

# 獲取最佳模型與參數
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_mse = grid_search.best_score_

# 輸出最佳參數與交叉驗證結果
print("最佳參數：", best_params)
print(f"最佳交叉驗證 MSE: {best_mse:.3f}")

# 使用最佳模型進行訓練和測試
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=25)
# best_model.fit(X_train, Y_train)

# 預測與評估
Y_train_pred = best_model.predict(X_train)
Y_test_pred = best_model.predict(X_test)
Y_total_pred = best_model.predict(X)

train_r2 = r2_score(Y_train, Y_train_pred)
train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
train_mae = mean_absolute_error(Y_train, Y_train_pred)
test_r2 = r2_score(Y_test, Y_test_pred)
test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
test_mae = mean_absolute_error(Y_test, Y_test_pred)
total_r2 = r2_score(Y, Y_total_pred)
total_rmse = np.sqrt(mean_squared_error(Y, Y_total_pred))
total_mae = mean_absolute_error(Y, Y_total_pred)



print(f"Train R²: {train_r2:.3f}, RMSE: {train_rmse:.3f}")
print(f"Test R²: {test_r2:.3f}, RMSE: {test_rmse:.3f}")

# 預測值與實際值對比(0.8+0.2)
plt.figure(figsize=(10, 5))
plt.scatter(Y_train, Y_train_pred, alpha=0.5, c='red', label=f'Train Data   R²: {train_r2:.3f}   RMSE: {train_rmse:.3f}    MAE: {train_mae:.3f}')
plt.scatter(Y_test, Y_test_pred, alpha=0.5, c='blue', label=f'Test Data     R²: {test_r2:.3f}   RMSE: {test_rmse:.3f}   MAE: {test_mae:.3f}')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='black', linestyle='--', label='45° Reference Line')
plt.xlabel('Actual Overpotential')
plt.ylabel('Predicted Overpotential')
plt.title('Predicted vs Actual Overpotential')
plt.grid(True)
plt.legend()
plt.show()

# 預測值與實際值對比(total)
plt.figure(figsize=(10, 5))
plt.scatter(Y, Y_total_pred, alpha=0.5, c='red', label=f'Total Data   R²: {total_r2:.3f}   RMSE: {total_rmse:.3f}    MAE: {total_mae:.3f}')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='black', linestyle='--', label='45° Reference Line')
plt.xlabel('Actual Overpotential')
plt.ylabel('Predicted Overpotential')
plt.title('Total Predicted vs Actual Overpotential')
plt.grid(True)
plt.legend()
plt.show()


# 輸出最佳特徵名稱
selected_features = best_model.named_steps['feature_selection'].get_support(indices=True)
poly_features = best_model.named_steps['poly'].get_feature_names_out(X.columns)
selected_feature_names = [poly_features[i] for i in selected_features]

print("最佳特徵名稱:")
print(selected_feature_names)

# 輸出模型的截距與係數
print('Intercept:', best_model.named_steps['lr'].intercept_)
coeff_df = pd.DataFrame(best_model.named_steps['lr'].coef_, selected_feature_names, columns=['Coefficient'])
print(coeff_df)


# 使用完整流水线处理特征数据
X_poly_transformed = best_model.named_steps['poly'].fit_transform(X)
poly_features = best_model.named_steps['poly'].get_feature_names_out(X.columns)  # 获取多项式特征名称

# 根据特征选择器筛选后的特征名称
selected_features_idx = best_model.named_steps['feature_selection'].get_support(indices=True)
selected_feature_names = [poly_features[i] for i in selected_features_idx]

# 获取特征选择后的数据
X_processed = best_model.named_steps['feature_selection'].transform(X_poly_transformed)

# 使用 SHAP 分析线性回归模型
lr_model = best_model.named_steps['lr']
explainer = shap.Explainer(lr_model, X_processed, feature_names=selected_feature_names)
shap_values = explainer(X_processed)

# 绘制 SHAP 图
shap.summary_plot(shap_values, X_processed, feature_names=selected_feature_names, plot_type="bar")
shap.summary_plot(shap_values, X_processed, feature_names=selected_feature_names)

# 分析单个数据点的 SHAP 解释
sample_idx = 0
shap.waterfall_plot(shap.Explanation(
    values=shap_values[sample_idx].values,
    base_values=shap_values[sample_idx].base_values,
    data=X_processed[sample_idx],
    feature_names=selected_feature_names))



# --------------------------------------------------------------------------------------------

from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(best_model,X,Y,cv=loo,scoring='neg_mean_squared_error')
print(np.sqrt(np.abs(scores)))

# Calculate RMSE for LOOCV
RMSE_LOOCV = np.sqrt(np.abs(scores).mean())

# Print the RMSE for LOOCV
print("Root Mean Squared Error (LOOCV):", RMSE_LOOCV)

#----------------------------------------------------------------------------------------------

# 外推法部分
X_test_df = pd.DataFrame(X_test, index=Y_test.index)
Y_test_df = Y_test.copy()

# 設定外推測試集，選取測試集中 OVER POTENTIAL 最小的 20%
top_20_percent_indices = Y_test_df.nsmallest(max(2, int(len(Y_test_df) * 0.25))).index
X_test_extrapolation = X_test_df.loc[top_20_percent_indices]
Y_test_extrapolation = Y_test_df.loc[top_20_percent_indices]

# 在外推測試集上進行預測
Y_pred_extrapolation = best_model.predict(X_test_extrapolation)

# 評估外推測試集的性能
if len(Y_test_extrapolation) >= 2:
    extrapolation_r2 = r2_score(Y_test_extrapolation, Y_pred_extrapolation)
    print(f"Extrapolation Test R²: {extrapolation_r2:.3f}")
else:
    extrapolation_r2 = None
    print("Extrapolation Test R²: Not calculated (insufficient samples)")

extrapolation_rmse = np.sqrt(mean_squared_error(Y_test_extrapolation, Y_pred_extrapolation))
print(f"Extrapolation Test RMSE: {extrapolation_rmse:.3f}")



# 定義每個變數的取值範圍
Temperature_range = np.arange(150, 201, 1)  # Temperature 從 150 到 200，以 1 為步長
Electrolyte_range = np.arange(0.1, 1.01, 0.01)  # Electrolyte 從 0 到 1，以 0.1 為步長
Ni_range = np.arange(0.35, 0.46, 0.01)  # Ni 的範圍從 0.35 到 0.8，步長為 0.05
Co_range = np.arange(0.35, 0.46, 0.01)  # Co 的範圍從 0.35 到 0.8，步長為 0.05
V_range = np.arange(0.1, 0.31, 0.01)  # V 的範圍從 0.1 到 0.5，步長為 0.05
Urea_range = np.arange(5.5, 7.6, 0.1)  # Urea 的範圍從 5.5 到 7.5，步長為 0.1

# 存儲所有有效組合
valid_combinations = []

# 計算 Ni 和 Co 的範圍上限，確保 V 的範圍符合設置（0.1 到 0.5）
max_Ni_Co = 0.5  # 這樣確保 1 - 2 * Ni_Co >= 0.1

  
# 遍歷 Temperature 和 Electrolyte 的所有可能值
for Temperature in Temperature_range:
    for Electrolyte in Electrolyte_range:
        for Ni_Co in Ni_range:
            if Ni_Co <= max_Ni_Co:  # 限制 Ni 和 Co 的範圍
                V = 1 - 2 * Ni_Co
                # 確保 V 在 0.1 到 0.5 的範圍內
                if 0.1 <= V <= 0.5 and Temperature > 0 and Electrolyte > 0 :
                    for Urea in Urea_range:
                         valid_combinations.append([Temperature, Electrolyte, Ni_Co, Ni_Co, V, Urea])


             

# 創建 DataFrame 存儲所有組合
combination_df = pd.DataFrame(valid_combinations, columns=['Temperature', 'Electrolyte', 'Ni', 'Co', 'V', 'Urea'])


# 使用训练时的 log_transformer 和 sqrt_transformer
combination_df['LOG_Temperature'] = np.log1p(combination_df['Temperature'].values)
combination_df['SQRT_Urea'] = np.sqrt(combination_df['Urea'].values)
combination_df = combination_df.drop(['Temperature', 'Urea'], axis=1)



# 确保列的顺序与训练数据一致
combination_df = combination_df[X.columns]


# 檢查新資料集中是否有缺失值
print("新資料集中缺失值統計：\n", combination_df.isnull().sum())


# 对新数据应用完整的 Pipeline 进行转换和预测
combination_df['Predicted_OVER_POTENTIAL'] = best_model.predict(combination_df)


# 構建原始特徵和預測值的資料框
combination_df_original_features = pd.DataFrame(valid_combinations, columns=['Temperature', 'Electrolyte', 'Ni', 'Co', 'V', 'Urea'])
combination_df_original_features['Predicted_OVER_POTENTIAL'] = combination_df['Predicted_OVER_POTENTIAL'].values

# 批次保存至 Excel
batch_size = 1000000  # 每個檔案保存的資料數量
output_folder = "D:/desktop/中興大學/實驗室/機器學習/完整/NiCoV"  # 修改為實際的保存路徑

# 確保目錄存在
import os
if not os.path.exists(output_folder):
        os.makedirs(output_folder)

# 將資料分批保存
for i in range(0, len(combination_df_original_features), batch_size):
    batch_df = combination_df_original_features.iloc[i:i+batch_size]
    batch_file_path = f"{output_folder}錢德魯_資料集(37 新)_全參數組合預測結果_批次_{i//batch_size + 1}.xlsx"
    batch_df.to_excel(batch_file_path, index=False)
    print(f"批次 {i//batch_size + 1} 保存完成：{batch_file_path}")

print("所有組合的預測結果已分批保存到 Excel 檔案中。")

# 找出最小值及其對應的特徵
min_value = combination_df_original_features['Predicted_OVER_POTENTIAL'].min()
min_value_row_idx = combination_df_original_features['Predicted_OVER_POTENTIAL'].idxmin()
min_value_row = combination_df_original_features.loc[min_value_row_idx]

# 顯示結果
print(f"最小的預測值: {min_value}")
print("對應的輸入特徵:")
print(min_value_row)    
