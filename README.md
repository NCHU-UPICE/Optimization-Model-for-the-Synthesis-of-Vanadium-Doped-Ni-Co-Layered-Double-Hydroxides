# Optimization-Model-for-the-Synthesis-of-Vanadium-Doped-Ni-Co-Layered-Double-Hydroxides
使用釩摻雜的鎳鈷層狀雙氫氧化合物之過電位資料集訓練本模型，透過隨機森林挑選權重較高的特徵進行多項式回歸模型的訓練，後續透過Pipeline與網格搜索，設置完整的模型訓練流程與最佳參數的尋找。訓練期間使用留一法交叉驗證( Leave-One-Out Cross-Validation)作為交叉驗證的方法，並得到RMSE=0.0162的低值。訓練完的模型展現出R2= 0.8422，表示其在預測過電位時的高準確性。使用此模型，大幅降低材料優化的難度與時間，為電催化觸媒的開發，拓展出一條嶄新的道路。
