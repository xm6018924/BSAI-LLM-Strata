"""百声AI Strata 启动器 - 启动 Strata 大模型服务"""
import os
import sys
import subprocess

def main():
    # 获取脚本所在目录的上级目录（ComfyUI 根目录）
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的路径
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Strata 启动脚本路径
    strata_bat = os.path.join(
        app_dir,
        "ComfyUI", "custom_nodes", "BSAI-LLM-Strata", "Strata", "run-q2_0.bat"
    )
    
    if not os.path.exists(strata_bat):
        print(f"错误：找不到 Strata 启动脚本")
        print(f"路径: {strata_bat}")
        input("按回车键退出...")
        return
    
    # 启动 Strata 服务
    strata_dir = os.path.dirname(strata_bat)
    print(f"正在启动百声AI Strata 服务...")
    print(f"工作目录: {strata_dir}")
    print()
    
    # 用新窗口启动
    subprocess.Popen(
        ["cmd", "/k", strata_bat],
        cwd=strata_dir,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

if __name__ == "__main__":
    main()
