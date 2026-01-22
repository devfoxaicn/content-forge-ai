#!/bin/bash
# AI Daily Digest - 一键生成简报并提交GitHub
# 快捷脚本

set -e

PROJECT_DIR="/Users/z/Documents/work/content-forge-ai"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python"

cd "$PROJECT_DIR"

echo "============================================================"
echo "🚀 AI Daily Digest - 一键生成简报并提交GitHub"
echo "============================================================"
echo ""

# 检查虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 虚拟环境不存在: $VENV_DIR"
    echo "请先创建虚拟环境: python -m venv venv"
    exit 1
fi

if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ Python解释器不存在: $PYTHON_BIN"
    exit 1
fi

echo "📦 使用虚拟环境: $VENV_DIR"
echo "🐍 Python: $PYTHON_BIN"
echo ""

# 运行Python脚本
PYTHONPATH="$PROJECT_DIR" "$PYTHON_BIN" "$PROJECT_DIR/scripts/daily_digest.py"

echo ""
echo "✨ 完成！"
