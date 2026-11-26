@echo off
cd /d "%~dp0"
title K-Means 聚类迭代模拟器
echo ============================================================
echo   K-Means 聚类迭代模拟器 · 正在启动
echo ============================================================
echo.

if not exist "jiemian.py" goto missing
if not exist "suanfa.py" goto missing

set "PYCMD=python"
python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('tkinter') else 1)" >nul 2>&1
if not errorlevel 1 goto launch

echo [自动切换] 当前 python 没有 tkinter，正在找其它 Python...
if exist "C:\python\python.exe" (
    set "PYCMD=C:\python\python.exe"
    goto launch
)
py -3 -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('tkinter') else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PYCMD=py -3"
    goto launch
)
goto no_python

:launch
echo [启动] %PYCMD%  运行  jiemian.py
echo 窗口关掉后本黑窗口会自动退出；万一报错会停在这里，方便截图。
echo.
%PYCMD% jiemian.py
if errorlevel 1 goto crashed
exit /b 0

:crashed
echo.
echo [异常] 程序非正常退出，上面的报错可以直接截图发到小组群里。
echo 常见原因：缺少 suanfa.py、tkinter 未安装、代码被改动。
echo.
pause
exit /b 1

:missing
echo [失败] 没找到 jiemian.py 或 suanfa.py
echo 请把「最终版」整个文件夹一起拷贝，不要只拷单个 .py 文件。
echo 当前目录： %cd%
echo.
pause
exit /b 1

:no_python
echo [失败] 没找到带 tkinter 的 Python
echo 安装 Python 时请勾选 tcl/tk and IDLE 与 Add Python to PATH。
echo 组长本机可在命令行运行： C:/python/python.exe jiemian.py
echo.
pause
exit /b 1
