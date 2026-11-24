[Setup]
AppId={{69935EF4-F9B2-41F5-AC5C-498FB6A8AEF2}
AppName=忆栈 MemStack
AppVersion=0.1.25
DefaultDirName={localappdata}\Programs\MemStack
DefaultGroupName=忆栈 MemStack
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=dist
OutputBaseFilename=MemStack-Setup-0.1.25-x64
ChangesAssociations=yes
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; 安装包自身的图标 + 卸载程序图标（都由 make_icons.py 生成）
SetupIconFile=packaging\icon.ico
UninstallDisplayIcon={app}\MemStack.exe
UninstallDisplayName=忆栈 MemStack
; 允许用户改安装目录（默认仍是 per-user 的 LOCALAPPDATA）
DisableDirPage=no

[Languages]
Name: "chinesesimplified"; MessagesFile: "packaging\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "dist\notices\*"; DestDir: "{app}\licenses"; Flags: ignoreversion
Source: "adapter_example.py"; DestDir: "{app}\examples"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "packaging\Reconnect.url"; DestDir: "{app}"; DestName: "重连忆栈.url"; Flags: ignoreversion
Source: "dist\MemStack\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; ── 开始菜单：程序项 + 卸载项放在同一个组里 ──────────────────────
Name: "{group}\忆栈 MemStack"; Filename: "{app}\MemStack.exe"; IconFilename: "{app}\MemStack.exe"; Comment: "打开忆栈 MemStack"
Name: "{group}\卸载 忆栈 MemStack"; Filename: "{uninstallexe}"; IconFilename: "{app}\MemStack.exe"
Name: "{group}\重连忆栈"; Filename: "{app}\MemStack.exe"; Parameters: "memstack://reconnect"; IconFilename: "{app}\MemStack.exe"

; ── 桌面快捷方式 ────────────────────────────────────────────────
;   用 {autodesktop}：all-users 装在公共桌面，per-user 装在自己桌面，
;   跟着安装模式走，不会出现"装了我这边、快捷方式给了所有人"。
;   Tasks: desktopicon —— 受安装向导里那个勾选框控制
Name: "{autodesktop}\忆栈 MemStack"; Filename: "{app}\MemStack.exe"; IconFilename: "{app}\MemStack.exe"; Comment: "打开忆栈 MemStack"; Tasks: desktopicon

; ── 快速启动栏（Win7 起已弱化，但老习惯还在，装一个不碍事）────────
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\忆栈 MemStack"; Filename: "{app}\MemStack.exe"; IconFilename: "{app}\MemStack.exe"

[Registry]
Root: HKCU; Subkey: "Software\Classes\memstack"; ValueType: string; ValueData: "URL:MemStack"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\memstack"; ValueType: string; ValueName: "URL Protocol"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\memstack\shell\open\command"; ValueType: string; ValueData: """{app}\MemStack.exe"" ""%1"""

[Tasks]
; 桌面图标默认勾上，用户可在安装向导里取消
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"; Flags: checkedonce

[Run]
Filename: "{app}\MemStack.exe"; Description: "打开忆栈 MemStack"; Flags: nowait postinstall skipifsilent

