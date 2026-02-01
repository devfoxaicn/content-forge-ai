#!/bin/bash
# ML Series批量生成脚本 - 并行度为3

# 设置项目根目录
export PYTHONPATH=/Users/z/Documents/work/content-forge-ai

# 配置文件
CONFIG_FILE="config/ml_topics_100_complete.json"
MAIN_SCRIPT="src/main.py"

# 日志目录
LOG_DIR="logs/batch_generate"
mkdir -p "$LOG_DIR"

# 待生成的episode列表（根据配置文件状态为pending的）
EPISODES=(
  56 57 58 59 60  # ml_series_6 (推荐系统) - 51-55已完成
  69 70  # ml_series_7 (模型优化) - 61-68已完成
  75 76 77 78  # ml_series_8 (传统机器学习) - 71-74已完成
  84 85 86 87 88 89  # ml_series_9 (特征工程) - 81-83已完成
  91 92 93 94 95 96 97 98 99 100  # ml_series_10 (高级ML专题)
)

# 并行度
PARALLELISM=3

# 当前运行的进程数
RUNNING=0

# 启动生成任务
start_episode() {
  local episode=$1
  local log_file="$LOG_DIR/episode_${episode}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 Episode $episode 生成..."
  PYTHONPATH=/Users/z/Documents/work/content-forge-ai python3 "$MAIN_SCRIPT" --mode series --series-config "$CONFIG_FILE" --episode "$episode" > "$log_file" 2>&1 &
  local pid=$!
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Episode $episode (PID: $pid) 已启动"

  # 将PID和episode映射保存到临时文件
  echo "$pid:$episode" >> /tmp/ml_series_batch_pids.tmp

  return $pid
}

# 检查并清理已完成的进程
check_completed() {
  if [ -f /tmp/ml_series_batch_pids.tmp ]; then
    # 读取当前运行的进程
    > /tmp/ml_series_running.tmp
    while IFS=':' read -r pid episode; do
      if ps -p "$pid" > /dev/null 2>&1; then
        echo "$pid:$episode" >> /tmp/ml_series_running.tmp
      else
        # 进程已结束，检查是否成功
        if [ -f "$LOG_DIR/episode_${episode}.log" ]; then
          if grep -q "✅.*生成完成" "$LOG_DIR/episode_${episode}.log"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Episode $episode 生成成功"
          else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Episode $episode 可能失败，请检查日志"
          fi
        fi
        RUNNING=$((RUNNING - 1))
      fi
    done < /tmp/ml_series_batch_pids.tmp

    mv /tmp/ml_series_running.tmp /tmp/ml_series_batch_pids.tmp
  fi
}

# 获取当前运行的进程数
get_running_count() {
  if [ -f /tmp/ml_series_batch_pids.tmp ]; then
    RUNNING=$(wc -l < /tmp/ml_series_batch_pids.tmp | tr -d ' ')
  else
    RUNNING=0
  fi
  echo $RUNNING
}

# 主循环
main() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== ML Series批量生成开始 =========="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 总共 ${#EPISODES[@]} 个episode待生成"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 并行度: $PARALLELISM"

  # 清空PID文件
  > /tmp/ml_series_batch_pids.tmp

  for episode in "${EPISODES[@]}"; do
    # 检查已完成的进程
    check_completed

    # 获取当前运行数
    get_running_count

    # 如果已达到并行度，等待
    while [ $RUNNING -ge $PARALLELISM ]; do
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] 当前运行 $RUNNING 个任务，等待中..."
      sleep 30
      check_completed
      get_running_count
    done

    # 启动新任务
    start_episode "$episode"
    RUNNING=$((RUNNING + 1))

    # 短暂延迟，避免同时启动过多
    sleep 5
  done

  # 等待所有任务完成
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有任务已启动，等待完成..."
  while [ $RUNNING -gt 0 ]; do
    check_completed
    get_running_count
    if [ $RUNNING -gt 0 ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] 剩余 $RUNNING 个任务运行中..."
      sleep 30
    fi
  done

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== ML Series批量生成完成 =========="
}

# 运行主函数
main
