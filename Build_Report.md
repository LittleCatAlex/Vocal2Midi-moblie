# Build_Report

**Status: The codebase is fully ported and ready for Cloud CI/CD Android builds!** 

## 架構調整 (ONNX Runtime Android 原生化)
為了徹底解決 `python-for-android` 缺乏 `onnxruntime` recipe 的限制，並捨棄手刻 C++ recipe 的高風險方案，我們改採用了**官方原生 Java 整合方案**：
1. **`OrtSessionWrapper` 實作：** 
   - 建立 `inference/ort_android_wrapper.py` 作為相容層。
   - 在 Android 環境下，透過 `pyjnius` 無縫呼叫官方 `ai.onnxruntime.OrtEnvironment` 等 Java 類別，並將 `numpy.ndarray` 轉換為 Java 的嵌套陣列餵入。
   - 處理了 `HashMap` 與 `OnnxTensor` 的生命週期 (close) 避免記憶體洩漏。
2. **零成本注入 (`sys.modules` Patching)：**
   - 為了不影響上層各種演算法 (GAME, HubertFA, TiFA, RMVPE) 的呼叫，我在 `main.py` 啟動時，使用 `sys.modules` 注入了自訂的 `OrtSessionWrapper` 與對應的常數 (如 `SessionOptions`, `GraphOptimizationLevel` 等)。
   - 這使得所有底層模組維持原有的 `import onnxruntime as ort` 不必修改，上層開發者體驗不變，而在手機上則會自動路由至 Java 原生引擎。
3. **構建配置更新 (`buildozer.spec`)：**
   - 成功自 Python requirements 中移除 `onnxruntime`。
   - 加入了 `pyjnius`。
   - 配置了 Gradle 原生相依：`android.gradle_dependencies = com.microsoft.onnxruntime:onnxruntime-android:latest.release`。

## 雲端自動化打包 (CI/CD) 資源
由於 Windows 沙盒本質限制，實體打包請交由以下提供的開箱即用工具：
1. **GitHub Actions:** 已於 `.github/workflows/build_apk.yml` 建立標準 Ubuntu 工作流，只要推送至 `main` 分支，即會自動編譯並上傳 APK 作為 Artifacts。
2. **Google Colab:** 已建立 `Colab_Build_Guide.md`，包含一鍵設定、安裝、打包與自動下載 APK 的 Jupyter 腳本。
