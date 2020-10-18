# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['try.py'],
             pathex=['D:\\eit226\\EITC_git\\EITC_ComputerLap\\venv2\\Lib\\site-packages', 'D:\\eit226\\EITC_git\\EITC_ComputerLap'],
             binaries=[],
             datas=[],
             hiddenimports=['numpy', 'PySimpleGUI', 'cv2', 'support', 'tensorflow', 'copy', 'threading', 'datetime'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='try',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
