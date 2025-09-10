from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFrame, QMessageBox, QCheckBox, QStackedWidget, QComboBox, QDateEdit, QTableView, QFileDialog, QSpinBox)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QPoint, QDate
from PyQt6.QtGui import QFont, QPixmap, QIcon, QPalette, QColor, QStandardItemModel, QStandardItem
import os
import pandas as pd
from datetime import datetime
import sys
from FrontEnd import DatabaseManager as FE_DB
from FrontEnd import api_client

class DataViewTab(QWidget):
    def __init__(self):
        super().__init__()
        self.current_page = 1
        self.total_pages = 1
        self.total_records = 0
        self.page_size = 100
        self.initUI()


        self.start_date = QDate.currentDate().addDays(-7)
        self.end_date = QDate.currentDate()
        self.symbol = "IF"
        
        
    def initUI(self):
        # 主布局
        layout = QVBoxLayout()
        
        # 顶部操作栏（导入/导出）
        top_bar = QHBoxLayout()
        self.import_btn = QPushButton("导入数据")
        self.export_btn = QPushButton("导出结果")
        top_bar.addWidget(self.import_btn)
        top_bar.addWidget(self.export_btn)
        top_bar.addStretch(1)
        layout.addLayout(top_bar)

        # 数据表格
        self.table_view = QTableView()
        self.model = QStandardItemModel()
        self.table_view.setModel(self.model)
        
        # 设置表格属性
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.table_view)
        
        # 分页控制栏
        page_control = QHBoxLayout()
        
        # 每页记录数选择
        page_control.addWidget(QLabel("每页显示:"))
        self.page_size_combo = QComboBox()
        self.page_size_combo.addItems(["50", "100", "200", "500"])
        self.page_size_combo.setCurrentIndex(1)  # 默认100条
        page_control.addWidget(self.page_size_combo)
        
        # 分页导航按钮
        self.first_page_btn = QPushButton("首页")
        self.prev_page_btn = QPushButton("上一页")
        self.next_page_btn = QPushButton("下一页")
        self.last_page_btn = QPushButton("末页")
        
        page_control.addWidget(self.first_page_btn)
        page_control.addWidget(self.prev_page_btn)
        page_control.addWidget(self.next_page_btn)
        page_control.addWidget(self.last_page_btn)
        
        # 页码显示
        page_control.addWidget(QLabel("页码:"))
        self.current_page_spin = QSpinBox()
        self.current_page_spin.setMinimum(1)
        self.current_page_spin.setMaximum(1)
        self.current_page_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.total_pages_label = QLabel("/ 1")
        page_control.addWidget(self.current_page_spin)
        page_control.addWidget(self.total_pages_label)
        
        page_control.addStretch(1)
        layout.addLayout(page_control)
    

        # 底部状态栏
        status_layout = QHBoxLayout()
        self.status_label = QLabel("就绪")
        self.record_count_label = QLabel("记录数: 0")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.record_count_label)
        
        layout.addLayout(status_layout)
        self.setLayout(layout)
    
        # 连接信号
        self.import_btn.clicked.connect(self.import_data)
        self.export_btn.clicked.connect(self.export_data)
        self.first_page_btn.clicked.connect(self.go_to_first_page)
        self.prev_page_btn.clicked.connect(self.go_to_prev_page)
        self.next_page_btn.clicked.connect(self.go_to_next_page)
        self.last_page_btn.clicked.connect(self.go_to_last_page)
        self.current_page_spin.editingFinished.connect(self.go_to_specific_page)
        self.page_size_combo.currentTextChanged.connect(self.page_size_changed)
        
    def load_data(self):
        """从后端 API 加载数据"""
        try:
            # 获取每页显示条数（当前页由按钮/回车事件维护）
            self.page_size = int(self.page_size_combo.currentText())

            start_str = self.start_date.toString("yyyy-MM-dd") + "T00:00:00"
            end_str = self.end_date.toString("yyyy-MM-dd") + "T23:59:59"
            symbol = self.symbol

            records = FE_DB.range_data(symbol=symbol if symbol else "IF", start_time=start_str, end_time=end_str)
            df = pd.DataFrame(records)

            # 简单分页：切片
            total = len(df)
            self.total_records = total
            self.total_pages = max(1, (total + self.page_size - 1) // self.page_size)
            self.update_pagination_controls()
            start_idx = (self.current_page - 1) * self.page_size
            end_idx = start_idx + self.page_size
            df = df.iloc[start_idx:end_idx]

            # 字段名兼容处理
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

            
            # 更新表格模型
            self.update_table_model(df)
            
            self.status_label.setText(f"加载完成，共 {len(df)} 条记录")
            self.record_count_label.setText(f"记录数: {len(df)}")
            
        except Exception as e:
            QMessageBox.warning(self, "查询错误", f"数据查询失败: {str(e)}")

    def set_date_range(self, start_date, end_date, symbol):
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol

    def update_table_model(self, df):
        """更新表格模型"""
        self.model.clear()
        
        # 设置表头
        headers = ['代码', '时间', '买价', '买单挂单量', '买单成交量', 
                  '卖单挂单量', '卖单成交量']
        self.model.setHorizontalHeaderLabels(headers)
        
        # 填充数据
        for _, row in df.iterrows():
            items = [
                QStandardItem(str(row.get('symbol', ''))),
                QStandardItem(str(row.get('date_time', ''))),
                QStandardItem(f"{float(row.get('bid_price', 0)):.4f}"),
                QStandardItem(str(row.get('bid_order_qty', ''))),
                QStandardItem(str(row.get('bid_executed_qty', ''))),
                QStandardItem(str(row.get('ask_order_qty', ''))),
                QStandardItem(str(row.get('ask_executed_qty', ''))),                
            ]
            self.model.appendRow(items)
        
    def export_data(self):
        """导出当前数据到 CSV / Excel（通过后端 API）"""
        start_str = self.start_date.toString("yyyy-MM-dd") + "T00:00:00"
        end_str = self.end_date.toString("yyyy-MM-dd") + "T23:59:59"
        contract_str= self.symbol or "all"

        # 1. 弹出保存对话框
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出数据",
            os.path.join(
                os.path.expanduser("~"),
                f"{contract_str}_{start_str}_{end_str}"
            ),
            "CSV 文件 (*.csv);;Excel 文件 (*.xlsx)"
        )
        if not file_path:        # 用户点了取消
            return

        # 2. 根据扩展名决定格式
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in {".csv", ".xlsx"}:   # 防手抖
            ext = ".csv" if "CSV" in selected_filter else ".xlsx"
            file_path += ext

        # 3. 从后端直接下载二进制流，避免前端本地拼装
        try:
            fmt = "xlsx" if ext == ".xlsx" else "csv"
            content = api_client.export_financial_data(self.symbol or "IF", start_str.replace(' ', 'T'), end_str.replace(' ', 'T'), fmt)
            with open(file_path, "wb") as f:
                f.write(content)
            QMessageBox.information(self, "导出成功", f"已保存到：\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出过程中发生错误：\n{str(e)}")

    def import_data(self):
        """从文件上传到后端导入"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择导入文件", os.path.expanduser("~"), "CSV/Excel (*.csv *.xlsx *.xls)"
        )
        if not file_path:
            return
        try:
            resp = api_client.import_financial_data(file_path)
            stats = resp.get("stats", {})
            errors = resp.get("errorRows", [])
            msg = f"导入完成\n插入: {stats.get('inserted',0)}\n重复: {stats.get('duplicates',0)}\n错误: {stats.get('errors',0)}"
            if errors:
                msg += f"\n错误示例: {errors[:3]}"
            QMessageBox.information(self, "导入结果", msg)
            self.load_data()
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"上传/导入失败：\n{str(e)}")



    def update_pagination_controls(self):
        """更新分页控件状态"""
        # 更新页码控件
        self.current_page_spin.blockSignals(True)  # 防止触发事件
        self.current_page_spin.setMaximum(self.total_pages)
        self.current_page_spin.setValue(self.current_page)
        self.current_page_spin.blockSignals(False)
        
        # 更新总页数标签
        self.total_pages_label.setText(f"/ {self.total_pages}")
        
        # 更新按钮状态
        self.prev_page_btn.setEnabled(self.current_page > 1)
        self.next_page_btn.setEnabled(self.current_page < self.total_pages)
        self.first_page_btn.setEnabled(self.current_page > 1)
        self.last_page_btn.setEnabled(self.current_page < self.total_pages)

    def go_to_first_page(self):
        """跳转到第一页"""
        self.current_page = 1
        self.load_data()

    def go_to_prev_page(self):
        """跳转到上一页"""
        if self.current_page > 1:
            self.current_page -= 1
            self.load_data()

    def go_to_next_page(self):
        """跳转到下一页"""
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.load_data()

    def go_to_last_page(self):
        """跳转到最后一页"""
        self.current_page = self.total_pages
        self.load_data()

    def go_to_specific_page(self):
        """跳转到指定页码"""
        page = self.current_page_spin.value()
        if 1 <= page <= self.total_pages:
            self.current_page = page
            self.load_data()

    def page_size_changed(self):
        """每页记录数改变时重置到第一页"""
        self.current_page = 1
        self.page_size = int(self.page_size_combo.currentText())
        self.load_data()

    def __del__(self):
        pass




if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序字体
    font = QFont("微软雅黑", 9)
    app.setFont(font)

    data_view_tab = DataViewTab()
    data_view_tab.show()

    sys.exit(app.exec())