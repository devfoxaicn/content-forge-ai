#!/bin/bash
# AI Daily Digest - 一键生成简报并提交GitHub
# 快捷脚本

set -e

PROJECT_DIR="/Users/z/Documents/work/content-forge-ai"
cd "$PROJECT_DIR"

echo "============================================================"
echo "🚀 AI Daily Digest - 一键生成简报并提交GitHub"
echo "============================================================"
echo ""

# 运行Python脚本
PYTHONPATH="$PROJECT_DIR" python "$PROJECT_DIR/scripts/daily_digest.py"

echo ""
echo "✨ 完成！"
