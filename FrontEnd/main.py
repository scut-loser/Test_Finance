import os
import sys
from PyQt6.QtWidgets import QApplication

try:
    # 兼容项目根目录运行：python -m FrontEnd.main
    from FrontEnd.DatabaseManager import configure  # type: ignore
    from FrontEnd.Login import LoginWindow  # type: ignore
except ImportError:
    # 兼容 FrontEnd 目录内运行
    from DatabaseManager import configure  # type: ignore
    from Login import LoginWindow  # type: ignore


def main():
    base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8080")
    configure(base_url=base_url)
    app = QApplication(sys.argv)
    win = LoginWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()