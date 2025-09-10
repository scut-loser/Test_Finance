# 导入对应的模块
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QLabel, QComboBox, QStackedWidget, QDoubleSpinBox, QPushButton, QTableWidget,QSpinBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QMessageBox
import pandas as pd
from PyQt6.QtCore import QDate
from datetime import datetime
from PyQt6.QtWidgets import QTableWidgetItem
from FrontEnd import DatabaseManager as FE_DB



# 不再定义本地 ORM 映射，统一走后端 API


class PredictionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.start_date = QDate.currentDate().addDays(-7)
        self.end_date = QDate.currentDate()
        self.symbol = "IF"
        self._data = pd.DataFrame()
        
    def initUI(self):
        # 主布局
        main_layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QVBoxLayout()
        
        # 固定唯一模型，无需选择
        control_panel.addWidget(QLabel("使用模型: 最终模型"))
        
        # 执行按钮
        self.run_btn = QPushButton("执行预测")
        control_panel.addWidget(self.run_btn)
        
        # 结果指标显示：分类与回归分区
        from PyQt6.QtWidgets import QGroupBox
        cls_group = QGroupBox("分类指标 (Anomaly Detection)")
        cls_layout = QVBoxLayout(cls_group)
        self.cls_table = QTableWidget(4, 1)
        self.cls_table.setHorizontalHeaderLabels(["值"])
        self.cls_table.setVerticalHeaderLabels(["最佳阈值", "准确率", "召回率", "F1分数"]) 
        self.cls_table.horizontalHeader().setStretchLastSection(True)
        cls_layout.addWidget(self.cls_table)
        control_panel.addWidget(cls_group)

        reg_group = QGroupBox("回归指标 (Prediction)")
        reg_layout = QVBoxLayout(reg_group)
        self.reg_table = QTableWidget(4, 1)
        self.reg_table.setHorizontalHeaderLabels(["值"]) 
        self.reg_table.setVerticalHeaderLabels(["MSE", "MAE", "MAPE(%)", "R2"]) 
        self.reg_table.horizontalHeader().setStretchLastSection(True)
        reg_layout.addWidget(self.reg_table)
        control_panel.addWidget(reg_group)
        
        control_panel.addStretch(1)
        
        # 右侧结果显示区域
        result_panel = QVBoxLayout()
        
        # 预测结果图表
        from PyQt6.QtWidgets import QGroupBox
        chart_group = QGroupBox("预测结果图表")
        chart_layout = QVBoxLayout(chart_group)
        self.result_figure = Figure(figsize=(8, 6))
        self.result_canvas = FigureCanvas(self.result_figure)
        chart_layout.addWidget(self.result_canvas)
        result_panel.addWidget(chart_group)
        
        # 异常详情表格（分组盒）
        anomaly_group = QGroupBox("检测到的异常")
        anomaly_layout = QVBoxLayout(anomaly_group)
        self.anomaly_table = QTableWidget(0, 5)
        self.anomaly_table.setHorizontalHeaderLabels(["期货代码", "预测时间", "预测类型","预测值", "置信度"])
        self.anomaly_table.horizontalHeader().setStretchLastSection(True)
        anomaly_layout.addWidget(self.anomaly_table)
        result_panel.addWidget(anomaly_group)

        # 轻量样式，接近第二张截图
        style = """
        QGroupBox { border: 1px solid #dcdcdc; border-radius: 6px; margin-top: 12px;}
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #333; }
        """
        self.setStyleSheet(style)
        
        # 添加到主布局
        main_layout.addLayout(control_panel, 1)
        main_layout.addLayout(result_panel, 2)
        
        self.setLayout(main_layout)
        
        # 连接信号
        #self.model_combo.currentIndexChanged.connect(self.params_stack.setCurrentIndex)
        self.run_btn.clicked.connect(self.run_prediction)
        
    
    def get_data(self):
        """通过后端 API 获取数据"""
        start_str = self.start_date.toString("yyyy-MM-dd") + "T00:00:00"
        end_str = self.end_date.toString("yyyy-MM-dd") + "T23:59:59"
        records = FE_DB.range_data(symbol=self.symbol, start_time=start_str, end_time=end_str)
        df = pd.DataFrame(records)
        if not df.empty:
            if 'dateTime' in df.columns:
                df['date_time'] = df['dateTime']
            if 'bidPrice' in df.columns:
                df['bid_price'] = df['bidPrice']
            if 'bidOrderQty' in df.columns:
                df['bid_order_qty'] = df['bidOrderQty']
            if 'bidExecutedQty' in df.columns:
                df['bid_executed_qty'] = df['bidExecutedQty']
            if 'askOrderQty' in df.columns:
                df['ask_order_qty'] = df['askOrderQty']
            if 'askExecutedQty' in df.columns:
                df['ask_executed_qty'] = df['askExecutedQty']
        return df
        
    def run_prediction(self):
        """执行预测并保存结果"""
        algo = "FINAL_MODEL"
        
        try:
            # 获取数据
            data = self.get_data()

            if data.empty:
                QMessageBox.warning(self, "数据警告", "未找到指定日期范围内的相关数据")
                return
            
            # 调用后端本地预测 API
            resp = FE_DB.predict_local(symbol=self.symbol, algorithm_name=algo, prediction_type="price_prediction")
            # 更新指标
            self.update_metrics(resp)
            predictions = [resp]
            
            # 更新界面显示
            self.update_results(predictions)
            
            QMessageBox.information(self, "预测完成", "预测执行完成，结果已保存到数据库")
            
        except Exception as e:
            QMessageBox.critical(self, "预测错误", f"预测执行失败: {str(e)}")

    # 不再需要保存，本地预测由后端落库

    def update_results(self, predictions):
        self.result_figure.clear()
        ax = self.result_figure.add_subplot(111)
        # 简单展示预测值点
        times = []
        values = []
        for p in predictions:
            t = p.get('prediction_time') or p.get('predictionTime')
            v = p.get('predicted_value') or p.get('predictedValue')
            if t and v is not None:
                times.append(str(t))
                values.append(float(v))
        if values:
            ax.plot(range(len(values)), values, marker='o')
            ax.set_title('预测结果')
        self.result_canvas.draw()

    def update_metrics(self, resp: dict):
        # 分类
        best_thr = resp.get('best_threshold') or (resp.get('result', {}) if isinstance(resp.get('result'), dict) else {}).get('best_threshold')
        acc = resp.get('accuracy') or (resp.get('result', {}) if isinstance(resp.get('result'), dict) else {}).get('accuracy')
        rec = resp.get('recall') or (resp.get('result', {}) if isinstance(resp.get('result'), dict) else {}).get('recall')
        f1 = resp.get('f1') or (resp.get('result', {}) if isinstance(resp.get('result'), dict) else {}).get('f1')
        vals_cls = [best_thr, acc, rec, f1]
        for i, v in enumerate(vals_cls):
            item = QTableWidgetItem("" if v is None else f"{float(v):.4f}")
            self.cls_table.setItem(i, 0, item)

        # 回归
        mse = resp.get('mse') or (resp.get('result', {}) if isinstance(resp.get('result'), dict) else {}).get('mse')
        mae = resp.get('mae') or (resp.get('result', {}) if isinstance(resp.get('result'), dict) else {}).get('mae')
        mape = resp.get('mape') or (resp.get('result', {}) if isinstance(resp.get('result'), dict) else {}).get('mape')
        r2 = resp.get('r2') or (resp.get('result', {}) if isinstance(resp.get('result'), dict) else {}).get('r2')
        vals_reg = [mse, mae, mape, r2]
        for i, v in enumerate(vals_reg):
            item = QTableWidgetItem("" if v is None else f"{float(v):.4f}")
            self.reg_table.setItem(i, 0, item)

    def execute_model_prediction(self, model_name, data):
        return []

    def set_date_range(self, start_date, end_date, symbol):
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol

    def load_data(self):
        """加载并缓存数据，供图表/预测等使用"""
        try:
            self._data = self.get_data()
        except Exception:
            self._data = pd.DataFrame()
