"""
GitHub自动发布工具
将每日简报自动提交并推送到GitHub仓库
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger


class GitHubPublisher:
    """GitHub自动发布器"""

    def __init__(self, repo_path: str = None):
        """
        初始化GitHub发布器

        Args:
            repo_path: Git仓库路径，默认为当前目录
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.repo_path = self.repo_path.absolute()

        # 验证是否是Git仓库
        if not self._is_git_repo():
            raise ValueError(f"路径 {self.repo_path} 不是Git仓库")

        logger.info(f"GitHub发布器初始化成功: {self.repo_path}")

    def _is_git_repo(self) -> bool:
        """检查是否是Git仓库"""
        git_dir = self.repo_path / ".git"
        return git_dir.exists() or (self.repo_path / ".git").exists()

    def _run_git_command(self, cmd: list, cwd: Path = None) -> subprocess.CompletedProcess:
        """
        执行Git命令

        Args:
            cmd: Git命令列表
            cwd: 工作目录

        Returns:
            subprocess.CompletedProcess
        """
        work_dir = cwd or self.repo_path
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result

    def check_git_status(self) -> Dict[str, Any]:
        """
        检查Git状态

        Returns:
            Dict: 包含分支、状态、未提交更改等信息
        """
        status = {
            "branch": "unknown",
            "has_changes": False,
            "untracked_files": [],
            "modified_files": [],
            "is_dirty": False
        }

        try:
            # 获取当前分支
            branch_result = self._run_git_command(["git", "branch", "--show-current"])
            if branch_result.returncode == 0:
                status["branch"] = branch_result.stdout.strip()

            # 获取状态信息
            status_result = self._run_git_command(["git", "status", "--porcelain"])
            if status_result.returncode == 0:
                output = status_result.stdout.strip()
                if output:
                    status["has_changes"] = True
                    status["is_dirty"] = True

                    for line in output.split('\n'):
                        if line.startswith('??'):
                            status["untracked_files"].append(line[3:].strip())
                        elif line.startswith(' M') or line.startswith('M'):
                            status["modified_files"].append(line[3:].strip())

        except Exception as e:
            logger.error(f"检查Git状态失败: {e}")

        return status

    def add_and_commit(
        self,
        files: list,
        commit_message: str,
        author_name: str = "ContentForge AI",
        author_email: str = "contentforge@ai"
    ) -> bool:
        """
        添加文件并提交

        Args:
            files: 要添加的文件列表
            commit_message: 提交信息
            author_name: 作者名
            author_email: 作者邮箱

        Returns:
            bool: 是否成功
        """
        try:
            # 添加文件
            for file_path in files:
                file_path = Path(file_path)
                if file_path.is_absolute():
                    # 转换为相对于仓库根目录的路径
                    rel_path = file_path.relative_to(self.repo_path)
                else:
                    rel_path = file_path

                result = self._run_git_command(["git", "add", str(rel_path)])
                if result.returncode != 0:
                    logger.error(f"添加文件失败 {rel_path}: {result.stderr}")
                    return False

            # 提交
            commit_cmd = ["git", "commit", "-m", commit_message]
            commit_cmd.extend(["--author", f"{author_name} <{author_email}>"])

            result = self._run_git_command(commit_cmd)
            if result.returncode != 0:
                logger.error(f"提交失败: {result.stderr}")
                return False

            logger.info(f"Git提交成功: {commit_message}")
            return True

        except Exception as e:
            logger.error(f"添加并提交失败: {e}")
            return False

    def push(self, remote: str = "origin", branch: str = None) -> bool:
        """
        推送到远程仓库

        Args:
            remote: 远程仓库名称
            branch: 分支名，默认为当前分支

        Returns:
            bool: 是否成功
        """
        try:
            if branch is None:
                branch_result = self._run_git_command(["git", "branch", "--show-current"])
                if branch_result.returncode != 0:
                    logger.error("获取当前分支失败")
                    return False
                branch = branch_result.stdout.strip()

            logger.info(f"推送到远程: {remote}/{branch}")

            result = self._run_git_command(["git", "push", remote, branch])
            if result.returncode != 0:
                logger.error(f"推送失败: {result.stderr}")
                return False

            logger.info(f"推送成功: {remote}/{branch}")
            return True

        except Exception as e:
            logger.error(f"推送失败: {e}")
            return False

    def publish_daily_digest(
        self,
        digest_file: str,
        json_file: str = None,
        remote: str = "origin",
        branch: str = None
    ) -> bool:
        """
        发布每日简报到GitHub

        Args:
            digest_file: 简报Markdown文件路径
            json_file: 简报JSON文件路径（可选）
            remote: 远程仓库名称
            branch: 分支名

        Returns:
            bool: 是否成功
        """
        try:
            today = datetime.now().strftime("%Y年%m月%d日")

            # 构建提交信息
            commit_message = f"docs: AI每日热点 · {today}\n\n"
            commit_message += "🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n"
            commit_message += "Co-Authored-By: Claude <noreply@anthropic.com>"

            # 添加文件
            files_to_add = [digest_file]
            if json_file:
                files_to_add.append(json_file)

            # 提交
            if not self.add_and_commit(files_to_add, commit_message):
                return False

            # 推送
            if not self.push(remote, branch):
                return False

            logger.info(f"每日简报发布成功: {today}")
            return True

        except Exception as e:
            logger.error(f"发布每日简报失败: {e}")
            return False

    def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main"
    ) -> bool:
        """
        创建Pull Request（需要gh CLI工具）

        Args:
            title: PR标题
            body: PR描述
            head_branch: 源分支
            base_branch: 目标分支

        Returns:
            bool: 是否成功
        """
        try:
            # 检查gh命令是否可用
            check_gh = subprocess.run(["which", "gh"], capture_output=True)
            if check_gh.returncode != 0:
                logger.warning("gh CLI未安装，跳过创建PR")
                return False

            # 创建PR
            pr_cmd = [
                "gh", "pr", "create",
                "--title", title,
                "--body", body,
                "--base", base_branch,
                "--head", head_branch
            ]

            result = subprocess.run(pr_cmd, cwd=self.repo_path, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"创建PR失败: {result.stderr}")
                return False

            logger.info(f"PR创建成功: {title}")
            return True

        except Exception as e:
            logger.error(f"创建PR失败: {e}")
            return False


def create_daily_digest_commit(
    digest_content: str,
    digest_date: str = None,
    json_content: str = None
) -> bool:
    """
    便捷函数：创建并提交每日简报

    Args:
        digest_content: 简报内容
        digest_date: 简报日期，格式YYYYMMDD
        json_content: JSON格式内容（可选）

    Returns:
        bool: 是否成功
    """
    try:
        from src.utils.storage_v2 import StorageFactory

        # 获取存储实例
        storage = StorageFactory.create_daily()

        # 保存文件
        digest_file = storage.save_markdown("digest", f"digest_{digest_date or datetime.now().strftime('%Y%m%d')}.md", digest_content)

        if json_content:
            json_file = storage.save_json("digest", f"digest_{digest_date or datetime.now().strftime('%Y%m%d')}.json", json_content)
        else:
            json_file = None

        # 发布到GitHub
        publisher = GitHubPublisher()
        return publisher.publish_daily_digest(digest_file, json_file)

    except Exception as e:
        logger.error(f"创建每日简报提交失败: {e}")
        return False


if __name__ == "__main__":
    # 测试代码
    publisher = GitHubPublisher()
    status = publisher.check_git_status()
    print(f"Git状态: {status}")
