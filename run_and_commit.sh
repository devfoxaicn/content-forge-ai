#!/bin/bash

# ContentForge AI - 自动运行并提交到GitHub脚本
# 每天早上3点通过cron调用

set -e  # 遇到错误立即退出

# 项目路径
PROJECT_DIR="/Users/z/Documents/work/content-forge-ai"
cd "$PROJECT_DIR" || exit 1

# Python虚拟环境路径
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

# 日志文件
LOG_FILE="$PROJECT_DIR/logs/cron_$(date +%Y%m%d).log"
DATA_DIR="$PROJECT_DIR/data"

# 记录开始时间
echo "==========================================" | tee -a "$LOG_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# 1. 拉取最新代码（避免冲突）
echo "[1/4] 拉取GitHub最新代码..." | tee -a "$LOG_FILE"
git fetch origin main || git fetch origin master || true
git pull origin main || git pull origin master || echo "无法拉取，可能没有远程分支" | tee -a "$LOG_FILE"

# 2. 运行主程序（使用虚拟环境Python）
echo "[2/4] 开始运行ContentForge AI..." | tee -a "$LOG_FILE"
PYTHONPATH="$PROJECT_DIR" "$VENV_PYTHON" "$PROJECT_DIR/src/main.py" --once --workflow auto 2>&1 | tee -a "$LOG_FILE"

# 3. 检查是否有新的数据生成
echo "[3/4] 检查新生成的数据..." | tee -a "$LOG_FILE"
if [ -d "$DATA_DIR" ]; then
    # 强制添加data目录（即使被.gitignore忽略）
    git add -f data/ || true

    # 检查是否有暂存的更改
    if git diff --cached --quiet; then
        echo "没有新的数据需要提交" | tee -a "$LOG_FILE"
    else
        # 4. 提交到GitHub
        echo "[4/4] 提交数据到GitHub..." | tee -a "$LOG_FILE"
        CURRENT_DATE=$(date +%Y-%m-%d)
        git commit -m "feat: AI内容自动生成 - $CURRENT_DATE

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>" | tee -a "$LOG_FILE"

        # 推送到GitHub
        git push origin main || git push origin master | tee -a "$LOG_FILE"
        echo "✅ 数据已成功提交到GitHub!" | tee -a "$LOG_FILE"
    fi
else
    echo "⚠️  data目录不存在，跳过提交" | tee -a "$LOG_FILE"
fi

# 记录结束时间
echo "==========================================" | tee -a "$LOG_FILE"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
