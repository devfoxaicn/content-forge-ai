# ContentForge AI 部署指南

本文档详细说明如何在不同环境中部署和运行 ContentForge AI。

---

## 目录

- [系统要求](#系统要求)
- [本地部署](#本地部署)
- [定时任务配置](#定时任务配置)
- [云服务部署](#云服务部署)
- [Docker部署（可选）](#docker部署可选)
- [故障排查](#故障排查)

---

## 系统要求

### 硬件要求
- **CPU**: 2核心及以上
- **内存**: 2GB及以上可用内存
- **存储**: 500MB及以上可用空间
- **网络**: 稳定的互联网连接

### 软件要求
- **操作系统**: Linux / macOS / Windows (WSL2)
- **Python**: 3.10 或更高版本
- **Git**: 用于克隆代码仓库

---

## 本地部署

### 1. 克隆项目

```bash
git clone https://github.com/Ming-H/content-forge-ai.git
cd content-forge-ai
```

### 2. 创建虚拟环境

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (WSL2 或 PowerShell)
python -m venv venv
venv\Scripts\activate
```

### 3. 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 或安装完整依赖（包含可选功能）
pip install -r requirements_core.txt
```

### 4. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的API密钥
# 必需的密钥：
# - ZHIPUAI_API_KEY: 智谱AI密钥（https://open.bigmodel.cn/）
# - TAVILY_API_KEY: Tavily搜索密钥（https://tavily.com/）
```

### 5. 测试运行

```bash
# 运行一次完整工作流
PYTHONPATH=/path/to/content-forge-ai python src/main.py --once --workflow auto
```

### 6. 验证输出

检查 `data/` 目录是否生成了相应的内容：
- `digest/` - AI热点简报
- `longform/` - 长文本技术文章
- `xiaohongshu/` - 小红书笔记
- `twitter/` - Twitter帖子

---

## 定时任务配置

### Linux/macOS - 使用 cron

#### 编辑 crontab

```bash
crontab -e
```

#### 添加定时任务

```bash
# 每天早上3点运行
0 3 * * * /path/to/content-forge-ai/run_and_commit.sh

# 或每天早上8点和晚上8点各运行一次
0 8,20 * * * /path/to/content-forge-ai/run_and_commit.sh
```

#### 查看日志

```bash
# 查看今天的日志
cat logs/cron_$(date +%Y%m%d).log

# 实时查看日志
tail -f logs/cron_$(date +%Y%m%d).log
```

### Linux - 使用 systemd（推荐）

#### 创建服务文件

```bash
sudo nano /etc/systemd/system/content-forge-ai.service
```

#### 服务配置

```ini
[Unit]
Description=ContentForge AI Automation Service
After=network.target

[Service]
Type=oneshot
User=your_username
WorkingDirectory=/path/to/content-forge-ai
ExecStart=/path/to/content-forge-ai/venv/bin/python src/main.py --once --workflow auto
StandardOutput=append:/path/to/content-forge- logs/systemd.log
StandardError=append:/path/to/content-forge-ai/logs/systemd.log

[Install]
WantedBy=multi-user.target
```

#### 创建定时器

```bash
sudo nano /etc/systemd/system/content-forge-ai.timer
```

```ini
[Unit]
Description=ContentForge AI Daily Timer
Requires=content-forge-ai.service

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

#### 启用服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable content-forge-ai.timer
sudo systemctl start content-forge-ai.timer

# 查看状态
sudo systemctl status content-forge-ai.timer
sudo systemctl list-timers
```

### Windows - 使用任务计划程序

1. 打开"任务计划程序"
2. 创建基本任务
3. 设置触发器（如每天早上3点）
4. 设置操作：运行脚本
   - 程序：`C:\path\to\venv\Scripts\python.exe`
   - 参数：`src/main.py --once --workflow auto`
   - 起始于：`C:\path\to\content-forge-ai`

---

## 云服务部署

### 腾讯云/阿里云 服务器部署

#### 1. 服务器配置建议
- **CPU**: 2核心
- **内存**: 4GB
- **系统**: Ubuntu 22.04 LTS
- **存储**: 40GB SSD

#### 2. 安装系统依赖

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

#### 3. 部署应用

```bash
# 克隆代码
cd /opt
sudo git clone https://github.com/Ming-H/content-forge-ai.git
sudo chown -R $USER:$USER content-forge-ai
cd content-forge-ai

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
nano .env  # 填入API密钥

# 测试运行
PYTHONPATH=/opt/content-forge-ai python src/main.py --once --workflow auto
```

#### 4. 配置定时任务

```bash
# 编辑crontab
crontab -e

# 添加任务
0 3 * * * cd /opt/content-forge-ai && /opt/content-forge-ai/venv/bin/python src/main.py --once --workflow auto >> /opt/content-forge-ai/logs/cron.log 2>&1
```

---

## Docker部署（可选）

### 创建 Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p data logs

# 设置环境变量
ENV PYTHONPATH=/app

# 运行应用
CMD ["python", "src/main.py", "--once", "--workflow", "auto"]
```

### 构建和运行

```bash
# 构建镜像
docker build -t content-forge-ai .

# 运行容器
docker run -d \
  --name content-forge-ai \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  content-forge-ai
```

---

## 故障排查

### 1. API密钥错误

**症状**: 提示"请设置环境变量 XXX_API_KEY"

**解决方案**:
```bash
# 检查环境变量是否设置
echo $ZHIPUAI_API_KEY

# 确认 .env 文件存在且包含密钥
cat .env

# 重启终端或重新激活虚拟环境
```

### 2. 虚拟环境未激活

**症状**: Python模块找不到

**解决方案**:
```bash
# 确认虚拟环境已激活
which python

# 应该显示: /path/to/venv/bin/python

# 如果显示系统Python，重新激活
source venv/bin/activate
```

### 3. 依赖包冲突

**症状**: ImportError 或版本冲突

**解决方案**:
```bash
# 重新安装依赖
pip install --upgrade -r requirements.txt

# 或创建新的虚拟环境
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. 定时任务未执行

**症状**: cron任务没有运行

**解决方案**:
```bash
# 检查cron服务状态
sudo systemctl status cron

# 查看cron日志
grep CRON /var/log/syslog

# 确认脚本有执行权限
ls -l run_and_commit.sh

# 手动运行脚本测试
./run_and_commit.sh
```

### 5. 网络连接问题

**症状**: API调用超时或失败

**解决方案**:
```bash
# 测试网络连接
ping open.bigmodel.cn

# 检查防火墙设置
sudo ufw status

# 如果使用代理，设置环境变量
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

---

## 性能优化建议

### 1. 减少API调用成本

- 使用 `glm-4-flash` 替代 `glm-4.7`（成本降低约80%）
- 减少数据源数量（如仅保留hackernews + arxiv）
- 降低 `max_tokens` 设置

### 2. 提高运行速度

- 并行执行独立Agent
- 使用mock模式进行开发测试
- 减少数据源数量

### 3. 监控和维护

- 定期清理旧日志文件（保留最近7天）
- 监控data目录大小
- 定期更新依赖包

---

## 安全建议

1. **不要将 .env 文件提交到Git仓库**
   ```bash
   # 确认 .gitignore 包含 .env
   grep .env .gitignore
   ```

2. **定期轮换API密钥**
   - 每月更换一次API密钥
   - 使用不同的密钥用于开发和生产

3. **限制API访问权限**
   - 为API密钥设置IP白名单
   - 限制API调用的速率限制

4. **备份重要数据**
   - 定期备份 data 目录
   - 备份配置文件

---

## 更新和升级

### 更新代码

```bash
# 拉取最新代码
git pull origin main

# 更新依赖
pip install --upgrade -r requirements.txt

# 测试运行
PYTHONPATH=/path/to/content-forge-ai python src/main.py --once --workflow auto
```

---

## 获取帮助

- **GitHub Issues**: https://github.com/Ming-H/content-forge-ai/issues
- **项目文档**: [README.md](README.md)
- **API参考**: [API_REFERENCE.md](API_REFERENCE.md)

---

**最后更新**: 2026-01-09
**版本**: v2.3
