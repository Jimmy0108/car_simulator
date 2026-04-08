Improved Hybrid A* for Autonomous Parking System
基於改良型 Hybrid A 與多階段動態最佳化的自動停車路徑規劃模擬器*
📝 專案簡介 (Overview)
本專案實作了一套應用於自動駕駛車輛的全局路徑規劃系統 (Global Path Planner)，專為解決狹窄地下停車場等複雜非結構化環境而設計。
系統基於經典的 Hybrid A* 演算法進行深度改良，導入了非完整運動學約束 (Non-holonomic Constraints)、Voronoi 安全勢能場、雙重啟發式函數 (Dual Heuristics) 以及連續碰撞檢測 (CCD)。系統能依據真實車輛參數，針對四象限停車位生成平滑、無碰撞且符合物理極限的優越駕駛軌跡。
✨ 核心演算法與創新點 (Core Features & Innovations)
1. 雙重啟發式搜尋 (Dual Heuristic Search)
為解決傳統 A* 容易陷入局部最佳解（如 U 型死胡同）的問題，本系統綜合了兩種啟發式成本：
2D Dijkstra 演算法：包含障礙物膨脹 (Obstacle Inflation) 的網格搜尋，確保避開牆壁。
Reeds-Shepp (RS) 曲線：考慮車輛最小迴轉半徑的無障礙最短路徑。
數學方程式：
2. Voronoi 安全勢能場 (Voronoi Potential Field)
為解決路徑過於貼近障礙物的危險，系統在成本函數中加入了基於廣義 Voronoi 圖 (GVD) 的勢能評估。
系統會計算車身四個角落到障礙物的最短距離 ，當  小於有效半徑  時，施加排斥力：
勢能方程式：
3. 多階段動態最佳化 (Multi-stage Dynamic Optimization)
針對狹窄車道難以直接入庫的物理極限，系統動態在目標車位外的通道上採樣允許集合 (Admissible Set )，並計算最佳中繼點 (Waypoint)，將複雜任務拆解為「入口  中繼點  車位」兩階段規劃。
4. 運動學與連續碰撞防護 (Kinematics & CCD)
精準實作單車模型 (Bicycle Model)，以「後軸 (Rear Axle)」為座標原點計算轉向，並利用軸距 (Wheelbase) 進行真實幾何中心校正。
實作邊緣碰撞檢測 (Edge Collision Checking)，在步長 (Step Size) 之間插入中繼狀態 (Mid-state)，徹底消除彎道死角處的穿模效應 (Tunneling Effect)。
📂 專案檔案架構 (Project Structure)
整個系統採用高度模組化的 MVC 架構設計：
模組分類
檔案名稱
核心功能描述
核心設定
config.py
集中管理車輛物理尺寸、演算法權重與全動態參數化地圖的生成公式。
物理模型
models.py
定義車輛空間狀態 State(x, y, \theta)，處理後軸與車身四角的座標轉換。
環境感知
parking_map.py
將幾何邊界轉換為 0.5m 解析度的點雲障礙物矩陣，並計算歐氏距離。
安全防護
collision.py
提供實體碰撞檢測、邊界安全餘裕 (Clearance) 與高解析度 RS 曲線驗證。
成本評估
heuristics.py
實作 Voronoi 勢能計算與 2D Dijkstra 障礙物感知網格波。
演算法核
planner.py
包含單車運動學的節點擴展 (expand_nodes) 與改良版 Hybrid A* 搜尋邏輯。
後處理
smoothing.py
使用共軛梯度下降法 (CG) 拉平軌跡，並重新校正車頭切線角度。
情境生成
scenarios.py
計算並生成四象限的目標車位與相對應的進場姿態。
視覺化
visualization.py
利用 Matplotlib 進行高密度插值繪圖，渲染真實的連續車身掃掠體積。
執行入口
main.py / run_gui.py
終端機批次執行腳本與圖形化介面 (GUI) 啟動程式。

⚙️ 車輛參數設定 (Vehicle Parameters)
本專案已對齊真實車輛（如一般中型房車）的物理極限，參數可於 config.py 中自由調整：
車長 (Length): 4.410 m
車寬 (Width): 1.785 m
軸距 (Wheelbase): 2.650 m
最大前輪轉向角: 30.0 degrees
後軸轉彎半徑: 4.59 m
🚀 如何執行 (How to Run)
安裝依賴套件:
pip install numpy matplotlib scipy pyReedsShepp


終端機批次執行 (純文字介面):
python main.py


啟動圖形化介面 (GUI):
python run_gui.py


📊 預期輸出結果 (Expected Outputs)
系統會自動在 result/ 資料夾下生成高解析度的軌跡圖。
圖中包含：
黑色方塊：離散化的實體牆壁與車位線。
紅/橘色實虛線：代表車輛前進（紅實線）與倒車（橘虛線）的參考軌跡。
淺藍色半透明方框：高密度插值的車身掃掠體積，證明行駛過程的絕對安全。
綠色資訊框：顯示車輛最終停妥後，距離障礙物最近的 Top 3 危險點數據。
