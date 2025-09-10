import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFrame, QMessageBox, QCheckBox, QStackedWidget)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QPoint
from PyQt6.QtGui import QFont, QPixmap, QIcon, QPalette, QColor
import hashlib
import re
from PyQt6.QtWidgets import QScrollArea  
from .main_window import MainWindow
from FrontEnd import DatabaseManager as FE_DB

class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        # 前后端松耦合：不再直连数据库
        
    def setup_ui(self):
        # 窗口设置
        self.setWindowTitle("登录/注册")
        self.setFixedSize(900, 500)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 左侧图片区域
        left_frame = QFrame()
        left_frame.setObjectName("leftFrame")
        left_frame.setFixedWidth(400)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 添加图片或图标
        image_label = QLabel()
        image_label.setText("用户认证系统")
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet("font-size: 24px; color: white; font-weight: bold;")
        
        left_layout.addWidget(image_label)
        main_layout.addWidget(left_frame)
        
        # 右侧区域 - 使用堆叠窗口实现登录和注册页面切换
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setObjectName("rightFrame")
        
        # 登录页面
        self.login_page = self.create_login_page()
        # 注册页面
        self.register_page = self.create_register_page()
        
        self.stacked_widget.addWidget(self.login_page)
        self.stacked_widget.addWidget(self.register_page)
        
        main_layout.addWidget(self.stacked_widget)
        
        # 应用样式表
        self.apply_stylesheet()
        
    def create_login_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # 标题
        title_label = QLabel("用户登录")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 用户名输入框
        username_layout = QVBoxLayout()
        username_label = QLabel("用户名:")
        username_label.setObjectName("inputLabel")
        self.login_username_input = QLineEdit()
        self.login_username_input.setObjectName("usernameInput")
        self.login_username_input.setPlaceholderText("请输入用户名")
        self.login_username_input.setClearButtonEnabled(True)
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.login_username_input)
        
        # 密码输入框
        password_layout = QVBoxLayout()
        password_label = QLabel("密码:")
        password_label.setObjectName("inputLabel")
        self.login_password_input = QLineEdit()
        self.login_password_input.setObjectName("passwordInput")
        self.login_password_input.setPlaceholderText("请输入密码")
        self.login_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.login_password_input.setClearButtonEnabled(True)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.login_password_input)
        
        # 记住密码和忘记密码
        extra_layout = QHBoxLayout()
        self.remember_checkbox = QCheckBox("记住密码")
        self.remember_checkbox.setObjectName("rememberCheckbox")
        self.remember_checkbox.setChecked(True)
        forgot_button = QPushButton("忘记密码?")
        forgot_button.setObjectName("forgotButton")
        forgot_button.setFlat(True)
        forgot_button.clicked.connect(self.show_forgot_password_dialog)
        extra_layout.addWidget(self.remember_checkbox)
        extra_layout.addWidget(forgot_button)
        
        # 登录按钮
        login_button = QPushButton("登录")
        login_button.setObjectName("loginButton")
        login_button.clicked.connect(self.handle_login)
        
        # 注册链接
        signup_layout = QHBoxLayout()
        signup_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        signup_label = QLabel("没有账号?")
        signup_button = QPushButton("立即注册")
        signup_button.setObjectName("signupButton")
        signup_button.setFlat(True)
        signup_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        signup_layout.addWidget(signup_label)
        signup_layout.addWidget(signup_button)
        
        # 添加到布局
        layout.addWidget(title_label)
        layout.addSpacing(20)
        layout.addLayout(username_layout)
        layout.addLayout(password_layout)
        layout.addLayout(extra_layout)
        layout.addSpacing(10)
        layout.addWidget(login_button)
        layout.addLayout(signup_layout)
        
        return page
        
    def create_register_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 0, 40, 0)
        layout.setSpacing(5)
        
        # 标题
        title_label = QLabel("用户注册")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 用户名输入框
        username_layout = QVBoxLayout()
        username_label = QLabel("用户名:")
        username_label.setObjectName("inputLabel")
        self.reg_username_input = QLineEdit()
        self.reg_username_input.setObjectName("usernameInput")
        self.reg_username_input.setPlaceholderText("请输入用户名 (3-20个字符)")
        self.reg_username_input.setClearButtonEnabled(True)
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.reg_username_input)
        
        # 密码输入框
        password_layout = QVBoxLayout()
        password_label = QLabel("密码:")
        password_label.setObjectName("inputLabel")
        self.reg_password_input = QLineEdit()
        self.reg_password_input.setObjectName("passwordInput")
        self.reg_password_input.setPlaceholderText("请输入密码 (至少6个字符)")
        self.reg_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.reg_password_input.setClearButtonEnabled(True)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.reg_password_input)
        
        # 确认密码输入框
        confirm_password_layout = QVBoxLayout()
        confirm_password_label = QLabel("确认密码:")
        confirm_password_label.setObjectName("inputLabel")
        self.reg_confirm_password_input = QLineEdit()
        self.reg_confirm_password_input.setObjectName("passwordInput")
        self.reg_confirm_password_input.setPlaceholderText("请再次输入密码")
        self.reg_confirm_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.reg_confirm_password_input.setClearButtonEnabled(True)
        confirm_password_layout.addWidget(confirm_password_label)
        confirm_password_layout.addWidget(self.reg_confirm_password_input)
        
        # 邮箱输入框
        email_layout = QVBoxLayout()
        email_label = QLabel("邮箱:")
        email_label.setObjectName("inputLabel")
        self.reg_email_input = QLineEdit()
        self.reg_email_input.setObjectName("emailInput")
        self.reg_email_input.setPlaceholderText("请输入邮箱地址（可选）")
        self.reg_email_input.setClearButtonEnabled(True)
        email_layout.addWidget(email_label)
        email_layout.addWidget(self.reg_email_input)
        
        # 注册按钮
        register_button = QPushButton("注册")
        register_button.setObjectName("loginButton")
        register_button.clicked.connect(self.handle_register)
        
        # 返回登录链接
        back_layout = QHBoxLayout()
        back_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        back_label = QLabel("已有账号?")
        back_button = QPushButton("返回登录")
        back_button.setObjectName("signupButton")
        back_button.setFlat(True)
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        back_layout.addWidget(back_label)
        back_layout.addWidget(back_button)
        
        # 添加到布局
        layout.addWidget(title_label)
        layout.addSpacing(20)
        layout.addLayout(username_layout)
        layout.addLayout(password_layout)
        layout.addLayout(confirm_password_layout)
        layout.addLayout(email_layout)
        layout.addSpacing(10)
        layout.addWidget(register_button)
        layout.addLayout(back_layout)
        
        return page
        
    def apply_stylesheet(self):
        style = """
        #leftFrame {
            background-color: #4A90E2;
            border-top-left-radius: 10px;
            border-bottom-left-radius: 10px;
        }
        
        #rightFrame {
            background-color: #FFFFFF;
            border-top-right-radius: 10px;
            border-bottom-right-radius: 10px;
        }
        
        #titleLabel {
            font-size: 28px;
            font-weight: bold;
            color: #2C3E50;
            margin-bottom: 10px;
        }
        
        #inputLabel {
            font-size: 14px;
            color: #7F8C8D;
            margin-bottom: 5px;
        }
        
        QLineEdit {
            border: 1px solid #BDC3C7;
            border-radius: 5px;
            padding: 4px;
            font-size: 14px;
            background-color: #F9F9F9;
        }
        
        QLineEdit:focus {
            border: 1px solid #4A90E2;
            background-color: #FFFFFF;
        }
        
        #usernameInput, #passwordInput, #emailInput {
            border: 1px solid #BDC3C7;
            border-radius: 5px;
            padding: 4px;
            font-size: 14px;
            background-color: #F9F9F9;
        }
        
        #usernameInput:focus, #passwordInput:focus, #emailInput:focus {
            border: 1px solid #4A90E2;
            background-color: #FFFFFF;
        }
        
        #loginButton {
            background-color: #4A90E2;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
        }
        
        #loginButton:hover {
            background-color: #357ABD;
        }
        
        #loginButton:pressed {
            background-color: #2C5F9E;
        }
        
        #rememberCheckbox {
            color: #7F8C8D;
            font-size: 13px;
        }
        
        #forgotButton, #signupButton {
            color: #4A90E2;
            font-size: 13px;
            text-decoration: none;
        }
        
        #forgotButton:hover, #signupButton:hover {
            color: #357ABD;
            text-decoration: underline;
        }
        """
        self.setStyleSheet(style)
        
    # 彻底移除本地建表/连接逻辑，改为调用后端认证接口
            
    def hash_password(self, password):
        """使用SHA-256哈希密码"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_email(self, email):
        """验证邮箱格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def handle_login(self):
        """处理登录请求"""
        username = self.login_username_input.text().strip()
        password = self.login_password_input.text()
        
        if not username or not password:
            self.shake_window()
            QMessageBox.warning(self, "输入错误", "用户名和密码不能为空!")
            return
            
        try:
            resp = FE_DB.login(username, password)
            if resp.get("success"):
                QMessageBox.information(self, "登录成功", f"欢迎回来, {username}!")
                self.main_window = MainWindow()
                self.main_window.show()
                self.close()
            else:
                QMessageBox.warning(self, "登录失败", resp.get("message", "用户名或密码错误!"))
        except Exception as e:
            QMessageBox.critical(self, "登录错误", f"登录过程中出现错误：{e}")
            
    def handle_register(self):
        """处理注册请求"""
        username = self.reg_username_input.text().strip()
        password = self.reg_password_input.text()
        confirm_password = self.reg_confirm_password_input.text()
        email = self.reg_email_input.text().strip()
        
        # 验证输入
        if not username or not password or not confirm_password:
            QMessageBox.warning(self, "输入错误", "前3个字段必须填写!")
            return
            
        if len(username) < 3 or len(username) > 20:
            QMessageBox.warning(self, "输入错误", "用户名长度必须在3-20个字符之间!")
            return
            
        if len(password) < 6:
            QMessageBox.warning(self, "输入错误", "密码长度至少为6个字符!")
            return
            
        if password != confirm_password:
            QMessageBox.warning(self, "输入错误", "两次输入的密码不一致!")
            return
            
        if email and not self.validate_email(email):
            QMessageBox.warning(self, "输入错误", "邮箱格式不正确!")
            return
            
        try:
            resp = FE_DB.register(username, password)
            if resp.get("success"):
                QMessageBox.information(self, "注册成功", "账号创建成功，现在可以登录了!")
                self.stacked_widget.setCurrentIndex(0)
            else:
                QMessageBox.warning(self, "注册失败", resp.get("message", "请稍后重试"))
        except Exception as e:
            QMessageBox.critical(self, "注册错误", f"注册过程中出现错误：{e}")
            
    def show_forgot_password_dialog(self):
        """显示忘记密码对话框"""
        QMessageBox.information(self, "忘记密码", "请联系系统管理员重置密码。")
        
    def shake_window(self):
        """窗口抖动动画效果"""
        animation = QPropertyAnimation(self, b"pos")
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        animation.setDuration(200)
        animation.setLoopCount(2)
        
        current_pos = self.pos()
        animation.setKeyValueAt(0, current_pos)
        animation.setKeyValueAt(0.2, current_pos + QPoint(5, 0))
        animation.setKeyValueAt(0.4, current_pos + QPoint(-5, 0))
        animation.setKeyValueAt(0.6, current_pos + QPoint(5, 0))
        animation.setKeyValueAt(0.8, current_pos + QPoint(-5, 0))
        animation.setKeyValueAt(1, current_pos)
        
        animation.start()
        
    def closeEvent(self, event):
        """关闭窗口时断开数据库连接"""
        pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序字体
    font = QFont("微软雅黑", 9)
    app.setFont(font)
    
    login_window = LoginWindow()
    login_window.show()
    
    sys.exit(app.exec())