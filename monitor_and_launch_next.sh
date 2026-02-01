#!/bin/bash
# 监控当前任务并自动启动下一批episode

export PYTHONPATH=/Users/z/Documents/work/content-forge-ai

# 待生成的下一批episode (56-58)
NEXT_EPISODES=(56 57 58)
CONFIG_FILE="config/ml_topics_100_complete.json"

# 检查任务是否完成
is_task_complete() {
  local output_file=$1
  if grep -q "✅.*生成完成" "$output_file" 2>/dev/null; then
    return 0
  fi
  return 1
}

# 检查任务是否失败
is_task_failed() {
  local output_file=$1
  if grep -q "ERROR\|Exception\|Traceback" "$output_file" 2>/dev/null; then
    return 0
  fi
  return 1
}

# 启动新任务
launch_episode() {
  local episode=$1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 Episode $episode ..."
  PYTHONPATH=/Users/z/Documents/work/content-forge-ai python3 src/main.py --mode series --series-config "$CONFIG_FILE" --episode "$episode" > /tmp/episode_${episode}_output.log 2>&1 &
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Episode $episode (PID: $!) 已启动"
}

# 主监控循环
main() {
  local output_dir="/tmp/claude/-Users-z-Documents-work-content-forge-ai/tasks"
  local current_tasks=("b5345df" "b7ee62f" "b56ab3b")
  local task_names=("Episode 53" "Episode 54" "Episode 55")
  local completed_count=0
  local next_episode_index=0

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始监控现有任务..."
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 待启动的下一批: ${NEXT_EPISODES[@]}"

  while [ $completed_count -lt ${#current_tasks[@]} ]; do
    for i in "${!current_tasks[@]}"; do
      local task_id="${current_tasks[$i]}"
      local task_name="${task_names[$i]}"
      local output_file="$output_dir/$task_id.output"

      # 跳过已完成的任务
      if [ -f "/tmp/task_${i}_completed" ]; then
        continue
      fi

      # 检查是否完成
      if is_task_complete "$output_file"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $task_name 完成!"
        touch "/tmp/task_${i}_completed"
        completed_count=$((completed_count + 1))

        # 启动下一个episode
        if [ $next_episode_index -lt ${#NEXT_EPISODES[@]} ]; then
          local next_ep="${NEXT_EPISODES[$next_episode_index]}"
          launch_episode "$next_ep"
          next_episode_index=$((next_episode_index + 1))
        fi
      fi

      # 检查是否失败
      if is_task_failed "$output_file"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $task_name 可能失败，请检查日志"
        touch "/tmp/task_${i}_failed"
        completed_count=$((completed_count + 1))
      fi
    done

    sleep 30
  done

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有第一批任务已完成!"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 已启动 ${next_episode_index} 个新任务"

  # 清理标记文件
  rm -f /tmp/task_*_completed /tmp/task_*_failed
}

# 运行主函数
main
