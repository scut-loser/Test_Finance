# 导入对应的PyQt6模块
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QPushButton
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.dates as mdates
import pandas as pd
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import QDate
from datetime import datetime
from FrontEnd import DatabaseManager as FE_DB

class ChartViewTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.start_date = QDate.currentDate().addDays(-7)
        self.end_date = QDate.currentDate()
        self.symbol = "IF"
        

    def initUI(self):
        # 主布局
        layout = QVBoxLayout()
        
        # 控制面板
        control_panel = QHBoxLayout()
        
        # 图表类型选择
        self.chart_combo = QComboBox()
        self.chart_combo.addItems(["价格走势", "成交量", "持仓量", "价格+成交量", "异常检测结果"])
        control_panel.addWidget(QLabel("图表类型:"))
        control_panel.addWidget(self.chart_combo)
        
        # 显示异常点复选框
        self.show_anomalies = QCheckBox("显示异常点")
        self.show_anomalies.setChecked(True)
        control_panel.addWidget(self.show_anomalies)
        
        # 刷新按钮
        self.refresh_btn = QPushButton("更新图表")
        control_panel.addWidget(self.refresh_btn)
        
        control_panel.addStretch(1)
        layout.addLayout(control_panel)
        
        # 创建图表区域
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # 图表交互控件
        interactive_panel = QHBoxLayout()
        interactive_panel.addWidget(QLabel("缩放:"))
        
        self.zoom_in_btn = QPushButton("+")
        self.zoom_out_btn = QPushButton("-")
        self.reset_zoom_btn = QPushButton("重置")
        
        interactive_panel.addWidget(self.zoom_in_btn)
        interactive_panel.addWidget(self.zoom_out_btn)
        interactive_panel.addWidget(self.reset_zoom_btn)
        interactive_panel.addStretch(1)
        
        layout.addLayout(interactive_panel)
        self.setLayout(layout)
        
        # 连接信号
        self.refresh_btn.clicked.connect(self.update_chart)
        self.chart_combo.currentIndexChanged.connect(self.update_chart)
        self.show_anomalies.stateChanged.connect(self.update_chart)
        
    def update_chart(self):
        """更新图表数据"""
        try:
            start_str = self.start_date.toString("yyyy-MM-dd") + "T00:00:00"
            end_str = self.end_date.toString("yyyy-MM-dd") + "T23:59:59"

            records = FE_DB.range_data(symbol=self.symbol, start_time=start_str, end_time=end_str)
            if not records:
                self.figure.clear()
                self.canvas.draw()
                return

            # 后端字段是驼峰：dateTime, bidPrice, bidExecutedQty, askExecutedQty
            df = pd.DataFrame(records)
            # 兼容不同命名
            if 'date_time' in df.columns:
                df['date_time'] = pd.to_datetime(df['date_time'])
            elif 'dateTime' in df.columns:
                df['date_time'] = pd.to_datetime(df['dateTime'])
            else:
                raise ValueError('缺少时间字段')
            if 'bid_price' in df.columns:
                df['bid_price'] = pd.to_numeric(df['bid_price'], errors='coerce')
            elif 'bidPrice' in df.columns:
                df['bid_price'] = pd.to_numeric(df['bidPrice'], errors='coerce')
            if 'bid_executed_qty' in df.columns:
                df['bid_executed_qty'] = pd.to_numeric(df['bid_executed_qty'], errors='coerce')
            elif 'bidExecutedQty' in df.columns:
                df['bid_executed_qty'] = pd.to_numeric(df['bidExecutedQty'], errors='coerce')
            
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            chart_type = self.chart_combo.currentText()
            
            if chart_type == "价格走势":
                ax.plot(df['date_time'], df['bid_price'], label='买一价', linewidth=1)
                ax.set_ylabel('价格')
                
            elif chart_type == "成交量":
                ax.bar(df['date_time'], df['bid_executed_qty'], alpha=0.7, label='成交量')
                ax.set_ylabel('成交量')
                
            elif chart_type == "价格+成交量":
                ax2 = ax.twinx()
                ax.plot(df['date_time'], df['bid_price'], color='blue', label='价格', linewidth=0.8)
                ax2.bar(df['date_time'], df['bid_executed_qty'], color='orange', alpha=0.5, label='成交量', width=0.001)
                ax.set_ylabel('价格')
                ax2.set_ylabel('成交量')
            
            ax.set_xlabel('时间')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 格式化x轴日期显示
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            self.figure.autofmt_xdate()
            
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "图表错误", f"图表生成失败: {str(e)}")

    def set_date_range(self, start_date, end_date, symbol):
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序字体
    font = QFont("微软雅黑", 9)
    app.setFont(font)

    chart_view = ChartViewTab()
    chart_view.show()

    sys.exit(app.exec())