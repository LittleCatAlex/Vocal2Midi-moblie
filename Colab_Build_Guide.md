# Vocal2Midi - Google Colab 一鍵打包指南

由於本機端或 Windows 環境設定 Android 編譯工具鏈較為繁瑣，您可以使用免費的 Google Colab 環境進行雲端編譯。

## 使用步驟

1. 前往 [Google Colab](https://colab.research.google.com/) 並建立一個全新的筆記本 (New Notebook)。
2. 點擊選單列的「執行階段 (Runtime)」->「變更執行階段類型 (Change runtime type)」，硬體加速器選擇 `None` 或 `T4 GPU` 皆可，並儲存。
3. 複製下方提供的所有程式碼，貼上到一個新的程式碼儲存格 (Code Cell) 中。
4. 點擊儲存格左側的「播放 (Run)」按鈕開始執行。
5. 等待約 15~30 分鐘，程式會自動安裝環境、下載 SDK/NDK、打包 APK。
6. 完成後，APK 檔案會自動下載到您的電腦中。

---

## 一鍵編譯腳本 (請貼入 Colab 執行)

```python
# 1. 確保環境為 Ubuntu 並更新套件
!sudo apt-get update
!sudo apt-get install -y openjdk-17-jdk git zip unzip autoconf libtool pkg-config zlib1g-dev libncurses-dev cmake libffi-dev libssl-dev

# 2. 安裝 Python 相依套件
!pip install --upgrade buildozer cython virtualenv

# 3. 取得專案原始碼
# 請將此處的 URL 替換為您的 GitHub 專案網址 (若尚未上傳 GitHub，可直接將專案資料夾上傳至 Colab 的 /content/Vocal2Midi)
import os
if not os.path.exists('/content/Vocal2Midi'):
    !git clone https://github.com/YOUR_USERNAME/Vocal2Midi.git /content/Vocal2Midi

# 切換到專案目錄
%cd /content/Vocal2Midi

# 4. 執行 Buildozer 打包
# 使用 yes 自動同意 Android SDK 授權條款
!yes | buildozer android debug

# 5. 下載產出的 APK 檔案
from google.colab import files
import glob

apk_files = glob.glob('bin/*.apk')
if apk_files:
    print(f"打包成功！準備下載：{apk_files[0]}")
    files.download(apk_files[0])
else:
    print("找不到 APK 檔案，打包可能失敗，請檢查上方日誌。")
```
