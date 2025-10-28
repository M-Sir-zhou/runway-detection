@echo off
echo 正在安装跑道检测系统的依赖...
echo.

python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo 安装完成！
echo 现在可以运行 demo.py 来处理文件
echo.
pause


