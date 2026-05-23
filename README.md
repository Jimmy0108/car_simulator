# Improved Hybrid A* for Autonomous Parking System
基於改良型 Hybrid A* 與多階段動態最佳化的自動停車路徑規劃模擬器

## 📝 專案簡介 (Overview)
本專案實作了一套應用於自動駕駛車輛的全局路徑規劃系統 (Global Path Planner)，專為解決狹窄地下停車場等複雜非結構化環境而設計。
系統基於經典 Hybrid A* 進行深度改良，導入非完整運動學約束 (Non-holonomic Constraints)、Voronoi 安全勢能場、雙重啟發式函數 (Dual Heuristics) 與連續碰撞檢測 (CCD)。
系統能依據真實車輛參數，針對多象限停車位生成平滑、無碰撞且符合物理極限的駕駛軌跡。

## ✨ 核心演算法與創新點 (Core Features & Innovations)
1. **雙重啟發式搜尋 (Dual Heuristic Search)**
	- 2D Dijkstra 演算法：包含障礙物膨脹 (Obstacle Inflation) 的網格搜尋，確保避開牆壁。
	- Reeds-Shepp (RS) 曲線：考慮車輛最小迴轉半徑的無障礙最短路徑。
2. **Voronoi 安全勢能場 (Voronoi Potential Field)**
	- 成本函數中加入 GVD 勢能評估，避免路徑過度貼近障礙物。
3. **多階段動態最佳化 (Multi-stage Dynamic Optimization)**
	- 在目標車位外通道上採樣允許集合 (Admissible Set)，動態計算最佳中繼點 (Waypoint)，拆解為多階段規劃。
4. **運動學與連續碰撞防護 (Kinematics & CCD)**
	- 單車模型 (Bicycle Model) 與連續碰撞檢測，降低彎道穿模風險。
5. **可配置地圖與方向規則 (YAML Maps & Directional Rules)**
	- 支援 YAML 地圖描述與單向車道方向規則，可快速擴充場景。

## 📂 專案檔案架構 (Project Structure)
整個系統採用高度模組化的架構設計：

| 模組分類 | 檔案名稱 | 核心功能描述 |
| --- | --- | --- |
| 核心設定 | config.py | 車輛尺寸、權重、與參數策略設定。 |
| 物理模型 | models.py | 定義狀態 State(x, y, $\theta$) 與座標轉換。 |
| 地圖抽象 | base_map.py | 地圖抽象介面、方向規則入口、骨架中心線計算。 |
| 地圖實作 | parking_map.py | 內建停車場地圖與障礙物點集。 |
| 地圖載入 | map_loader.py | YAML 地圖載入器 (YamlParkingMap)。 |
| 方向規則 | map_rules.py | 走廊方向限制與 RS 路徑驗證。 |
| 成本評估 | heuristics.py | Voronoi 勢能、Dijkstra 網格與雙重啟發式。 |
| 演算法核 | planner.py | 經典 Hybrid A* 多階段搜尋流程。 |
| 演算法核 | planner_directional.py | 方向規則模式的多階段 Hybrid A*。 |
| 安全防護 | collision.py | 碰撞檢測、清障與 RS 曲線驗證。 |
| 後處理 | smoothing.py | 共軛梯度 (CG) 平滑與航向修正。 |
| 情境生成 | scenarios.py | 生成停車格位與進場姿態。 |
| 視覺化 | visualization.py | Matplotlib 繪圖與車身掃掠體積。 |
| GUI | gui.py / run_gui.py | 圖形化介面與啟動器。 |
| 執行入口 | main.py | 互動式地圖 / 模式選擇與批次規劃。 |
| 場景資料 | maps/*.yaml | YAML 地圖與方向規則設定。 |

## ⚙️ 車輛參數設定 (Vehicle Parameters)
參數可於 config.py 中調整：
- 車長 (Length): 4.410 m
- 車寬 (Width): 1.785 m
- 軸距 (Wheelbase): 2.650 m
- 最大前輪轉向角: 30.0 degrees
- 後軸轉彎半徑: 4.59 m

## 🚀 如何執行 (How to Run)
### 安裝依賴套件
```bash
pip install numpy matplotlib scipy pyyaml pyReedsShepp
```

### 終端機批次執行 (純文字介面)
```bash
python main.py
```
執行後會依序提示：
1. 地圖選擇 (內建或 YAML)
2. 規劃模式 (無規則 / 方向規則)
3. 停車格位選擇

### 啟動圖形化介面 (GUI)
```bash
python run_gui.py
```

## 🗺️ YAML 地圖格式 (Map Configuration)
YAML 主要欄位如下：
- metadata: 地圖名稱與版本
- layout: 地圖尺寸與解析度
- obstacles: 障礙物定義 (線段 / 矩形 / 離散點)
- visualization: 視覺化元素 (牆面、車格線、標籤)
- scenarios: 停車格位與外向角度設定
- start_state: 起始姿態
- directional_rules: 單向車道與驗證參數

## 📊 預期輸出結果 (Expected Outputs)
系統會在 result/ 生成高解析度軌跡圖，包含：
- 黑色方塊：離散化的牆壁與車位線。
- 紅 / 橘色實虛線：前進與倒車路徑。
- 淺藍色半透明方框：車身掃掠體積。
- 綠色資訊框：距離障礙物最近的 Top 3 危險點數據。
