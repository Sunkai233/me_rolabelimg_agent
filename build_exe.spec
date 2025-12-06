# -*- mode: python ; coding: utf-8 -*-
# 目录模式 - 最稳定的打包方式，解决所有 DLL 问题
import os
import sys

block_cipher = None

# 路径配置
venv_path = r'D:\roLabelImg-master\.venv'
conda_base = r'C:\Users\Panasonic\.conda\envs\yolo'

binaries_list = []

# ============ 添加基本 DLL ============
dll_sources = [
    os.path.join(conda_base, 'Library', 'bin'),
    os.path.join(conda_base, 'DLLs'),
    os.path.join(venv_path, 'Scripts'),
]

dll_names = [
    'libexpat.dll',
    'libcrypto-3-x64.dll',
    'libssl-3-x64.dll',
]

for dll_name in dll_names:
    for source_dir in dll_sources:
        dll_path = os.path.join(source_dir, dll_name)
        if os.path.exists(dll_path):
            binaries_list.append((dll_path, '.'))
            print(f"✓ {dll_name}")
            break

# ============ 手动添加 PyTorch 所有 DLL ============
print("\n🔧 手动添加 PyTorch DLL...")

try:
    import torch
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')

    if os.path.exists(torch_lib_path):
        torch_dlls_found = 0
        for file in os.listdir(torch_lib_path):
            if file.endswith('.dll'):
                dll_full_path = os.path.join(torch_lib_path, file)
                binaries_list.append((dll_full_path, '.'))  # 👈 放在根目录
                torch_dlls_found += 1
                if torch_dlls_found <= 5:
                    print(f"   ✓ {file}")

        if torch_dlls_found > 5:
            print(f"   ✓ ... 以及其他 {torch_dlls_found - 5} 个 DLL")

        print(f"✅ 共找到 {torch_dlls_found} 个 PyTorch DLL")
    else:
        print(f"⚠️ PyTorch lib 目录不存在: {torch_lib_path}")
except Exception as e:
    print(f"❌ 添加 PyTorch DLL 时出错: {e}")

# ============ 添加 MSVC 运行库 ============
print("\n🔧 查找 MSVC 运行库...")

msvc_dlls = [
    'msvcp140.dll',
    'vcruntime140.dll',
    'vcruntime140_1.dll',
    'msvcp140_1.dll',
    'msvcp140_2.dll',
]

system_paths = [
    r'C:\Windows\System32',
    r'C:\Windows\SysWOW64',
    os.path.join(conda_base, 'Library', 'bin'),
    os.path.join(venv_path, 'Scripts'),
]

for dll_name in msvc_dlls:
    found = False
    for sys_path in system_paths:
        dll_path = os.path.join(sys_path, dll_name)
        if os.path.exists(dll_path):
            binaries_list.append((dll_path, '.'))
            print(f"   ✓ {dll_name}")
            found = True
            break
    if not found:
        print(f"   ⚠️ 未找到: {dll_name}")

# ============ 数据文件 ============
datas_list = [
    ('data', 'data'),
    ('icons', 'icons'),
    ('libs', 'libs'),
    ('resources.py', '.'),
    ('result_viewer.py', '.'),
    ('config_dialog.py', '.'),
    ('xml_to_json_converter.py', '.'),
    ('meter_predictor.py', '.'),
    ('bj_inference_main.py', '.'),
]

# 添加 Python 模型文件
for model_file in ['zhizhen_model.py', 'youwei_model.py', 'shuxian_model.py']:
    if os.path.exists(model_file):
        datas_list.append((model_file, '.'))

# 添加字体
if os.path.exists('SimHei.ttf'):
    datas_list.append(('SimHei.ttf', '.'))

# ============ Analysis ============
a = Analysis(
    ['roLabelImg.py'],
    pathex=['D:\\roLabelImg-master'],
    binaries=binaries_list,
    datas=datas_list,
    hiddenimports=[
        # PyQt5
        'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets', 'PyQt5.QtPrintSupport',

        # XML
        'lxml', 'lxml.etree', 'xml.etree.ElementTree',

        # 新增模块
        'result_viewer', 'config_dialog', 'xml_to_json_converter',
        'meter_predictor', 'bj_inference_main',
        'zhizhen_model', 'youwei_model', 'shuxian_model',

        # PyTorch - 详细导入
        'torch', 'torch._C', 'torch._utils',
        'torch.nn', 'torch.nn.functional',
        'torch.backends', 'torch.backends.cudnn',
        'torchvision', 'torchvision.models',

        # Ultralytics
        'ultralytics', 'ultralytics.nn', 'ultralytics.utils',

        # 其他
        'cv2', 'PIL', 'numpy',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'pandas', 'paddleocr', 'paddle'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ============ EXE（目录模式，不包含 binaries）============
exe = EXE(
    pyz,
    a.scripts,
    [],                         # 👈 空的，不包含 binaries
    exclude_binaries=True,      # 👈 关键：排除二进制文件
    name='roLabelImg',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,               # 调试时 True，正式版 False
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icons/app.ico' if os.path.exists('icons/app.ico') else None,
)

# ============ COLLECT（收集所有文件到目录）============
coll = COLLECT(
    exe,
    a.binaries,                 # 👈 所有 DLL 在这里
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='roLabelImg'           # 👈 生成的文件夹名
)

print("\n" + "="*60)
print("📦 目录模式打包配置")
print("="*60)
print(f"✓ 总二进制文件数: {len(binaries_list)}")
print(f"✓ 总数据文件数: {len(datas_list)}")
print("="*60)
print("\n💡 提示：")
print("   目录模式会生成一个文件夹而不是单个 exe")
print("   文件夹中包含 exe 和所有 DLL")
print("   这是最稳定的打包方式！")
print("="*60 + "\n")