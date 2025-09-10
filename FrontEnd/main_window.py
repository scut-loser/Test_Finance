import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget,
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QTableView, QComboBox,QDateEdit,QMenu
)
from PyQt6.QtCore import Qt, QAbstractTableModel,QDate
from PyQt6.QtGui import QIcon,QAction
from PyQt6.QtGui import QFont, QPixmap, QIcon, QPalette, QColor
from .DataViewTab import DataViewTab
from .ChartViewTab import ChartViewTab
from .PredictionTab import PredictionTab
from . import DatabaseManager as FE_DB


class MainWindow(QMainWindow):                                                                          
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("期货交易异常检测系统")
        self.setGeometry(100, 100, 1200, 800)

        # 创建共享时间控件
        # self.start_date_edit = QDateEdit(QDate.currentDate().addDays(-7))
        # 测试用 开始时间
        self.start_date_edit = QDateEdit(QDate(2025,2,12))
        self.end_date_edit = QDateEdit(QDate.currentDate())

        self.start_date_edit.setCalendarPopup(True)
        self.end_date_edit.setCalendarPopup(True)

        # 创建顶部控制条
        top_layout = QHBoxLayout()
        self.contract_combo = QComboBox()
        try:
            symbols = FE_DB.symbols()
            if symbols:
                self.contract_combo.addItems(symbols)
            else:
                self.contract_combo.addItems(["IF", "SC", "AU"])  # fallback
        except Exception:
            self.contract_combo.addItems(["IF", "SC", "AU"])  # fallback
        top_layout.addWidget(QLabel("合约:"))
        top_layout.addWidget(self.contract_combo)
        top_layout.addWidget(QLabel("开始日期:"))
        top_layout.addWidget(self.start_date_edit)
        top_layout.addWidget(QLabel("结束日期:"))
        top_layout.addWidget(self.end_date_edit)

        self.refresh_btn = QPushButton("刷新数据")
        top_layout.addWidget(self.refresh_btn)

        top_layout.addStretch(1)
        
        # # 创建选项卡
        self.tabs = QTabWidget()
        # self.setCentralWidget(self.tabs)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.tabs)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 连接时间变化信号
        self.contract_combo.currentIndexChanged.connect(self.on_date_changed)
        self.start_date_edit.dateChanged.connect(self.on_date_changed)
        self.end_date_edit.dateChanged.connect(self.on_date_changed)
        self.refresh_btn.clicked.connect(self.on_date_changed)
        
        # 添加各个标签页
        self.data_tab = DataViewTab()
        self.chart_tab = ChartViewTab()
        self.prediction_tab = PredictionTab()
        
        self.tabs.addTab(self.data_tab, "数据展示")
        self.tabs.addTab(self.chart_tab, "图表分析")
        self.tabs.addTab(self.prediction_tab, "异常预测")
        
        # 创建菜单栏
        self.create_menu()

        # 首次进入时按当前合约与日期加载一次数据
        self.on_date_changed()

    def on_date_changed(self):
        """时间改变时通知两个 tab 更新"""
        start_date = self.start_date_edit.date()
        end_date = self.end_date_edit.date()
        symbol = self.contract_combo.currentText() or None

        # 同步到 DataViewTab
        self.data_tab.set_date_range(start_date, end_date, symbol)
        self.data_tab.load_data()

        # 同步到 ChartViewTab（你可以扩展 ChartViewTab 使用这些日期）
        self.chart_tab.set_date_range(start_date, end_date, symbol)
        self.chart_tab.update_chart()
        # 同步到 PredictionTab
        self.prediction_tab.set_date_range(start_date, end_date, symbol)
        self.prediction_tab.load_data()
        
    def create_menu(self):
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        import_menu = QMenu('导入数据', self)
        real_time_import_action = QAction('实时数据', self)
        import_menu.addAction(real_time_import_action)
        history_import_action = QAction('历史数据', self)
        # 绑定到数据页的导入逻辑（后续看要不要分开，或者只留一个）
        history_import_action.triggered.connect(self.data_tab.import_data)
        import_menu.addAction(history_import_action)
        file_menu.addMenu(import_menu)
        
        
        export_action = QAction('导出结果', self)
        export_action.setShortcut('Ctrl+S')
        file_menu.addAction(export_action)
        # 绑定到数据页的导出逻辑（同上）
        export_action.triggered.connect(self.data_tab.export_data)

        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        about_action = QAction('关于', self)
        help_menu.addAction(about_action)
        #about_action.triggered.connect(self.on_about_triggered)

        def on_real_time_import_triggered(self):
            # api实时获取，窗口界面需要改变
            pass

        def on_history_import_triggered(self):
            # 有csv，数据库，api三种方式导入历史数据，都需要先把数据加载到数据库
            pass

        def on_export_triggered(self):
            # 导出原始数据到csv
            pass

        def on_about_triggered(self):
            # 显示关于信息
            pass

        
            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序字体
    font = QFont("微软雅黑", 9)
    app.setFont(font)

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec())