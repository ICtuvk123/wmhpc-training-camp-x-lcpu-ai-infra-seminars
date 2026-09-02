# 编辑器诊断

请用 VS Code 单独打开 `assignment01` 目录。首次打开时安装工作区推荐扩展，并选择安装了项目依赖的 Python 解释器（通常是 `.venv`）。使用 Remote SSH/WSL 时，扩展和解释器都要在远程窗口中选择。

## 实时检查

- `.cu`/`.cuh` 会按 CUDA C++ 打开，由 NVIDIA Nsight VS Code Edition 与 Microsoft C/C++ 提供高亮、补全和错误波浪线。CUDA Toolkit 的头文件会从 `CUDA_PATH`、`CUDA_HOME` 或 `/usr/local/cuda` 查找。
- `.py` 由 Pylance 检查普通 Python 语法和基础类型；Triton、TileLang 没有单独的文件语法，它们都是 Python DSL。

Python 中的 `...` 本身是合法语法，所以填空尚未完成时不一定会出现红线；只有运行对应测试、让 DSL 编译器处理 kernel，才能判断这类错误。

## 编译级检查

打开命令面板，运行 `Tasks: Run Task`，再选择：

- `检查 CUDA：编译当前文件`：调用 `nvcc`，错误会进入 Problems 面板，不生成可执行文件。
- `检查 Python/Triton/TileLang：当前文件语法`：只检查 Python 语法，不导入 GPU 依赖。
- `检查 Triton：运行相关测试并编译 kernel`：通过测试触发 Triton JIT，能发现 shape、mask、指针和编译约束错误。
- `检查 TileLang：运行相关测试并编译 kernel`：需要 GPU 和 TileLang，通过测试触发 TileLang 编译。

注意：仓库中有故意不能编译的填空题，也有故意包含运行时 bug 的调试题。红线只表示当前源码/环境有问题，不代表编辑器应该自动给出修法。
