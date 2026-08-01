"""
工作区上下文 (WorkspaceContext)
===========================
仿照 pico/workspace.py 的设计，为 minicoder2 提供安全的工作区边界。

作用：
1. 限定工具只能操作工作区内的文件，防止 "../" 路径逃逸
2. 提供工作区摘要信息（仓库状态、分支、文档等）
3. 所有文件类工具统一通过 context.path() 解析路径
"""

import os
import hashlib
import json
import subprocess
import textwrap
from pathlib import Path


class WorkspaceContext:
    """
    WorkspaceContext（workspace.py）
    → 职责：工作区长什么样
    → 方法：text()、fingerprint()
    """

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self._repo_root = self._find_repo_root(self.root)

    @classmethod
    def build(cls, cwd: str = ".") -> "WorkspaceContext":
        """从当前目录构建一个工作区上下文。"""
        return cls(root=cwd)

    @staticmethod
    def _find_repo_root(start: Path) -> Path:
        """向上查找 .git 目录来确定仓库根目录。"""
        current = start.resolve()
        for path in [current, *current.parents]:
            if (path / ".git").exists():
                return path
        return start

    @property
    def repo_root(self) -> str:
        return str(self._repo_root)

    def path(self, raw_path: str) -> Path:
        """安全地解析用户输入的路径。
        
        规则：
        - 相对路径以 self.root 为基准进行拼接
        - 绝对路径检查是否在工作区之内
        - 所有路径解析后检查是否越界
        
        Args:
            raw_path: 用户输入的原始路径字符串
            
        Returns:
            解析后的绝对 Path 对象
            
        Raises:
            PermissionError: 如果路径解析后超出了工作区范围
        """
        p = Path(raw_path)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self.root / p).resolve()

        # 检查是否越界：所有解析后的路径必须在 root 之下
        root_str = os.path.normcase(str(self.root))
        resolved_str = os.path.normcase(str(resolved))
        if not resolved_str.startswith(root_str + os.sep) and resolved_str != root_str:
            raise PermissionError(
                f"路径越界! '{resolved}' 不在工作区 '{self.root}' 内"
            )
        return resolved

    def text(self) -> str:
        """生成工作区摘要文本（用于 prompt 上下文）。"""
        branch = self._git_branch()
        status = self._git_status()
        commits = self._git_recent_commits()

        return textwrap.dedent(
            f"""\
            Workspace:
            - root: {self.root.name}
            - branch: {branch}
            - status:
            {status}
            - recent_commits:
            {commits}
            """
        ).strip()

    def fingerprint(self) -> str:
        """生成工作区指纹，用于检测工作区状态是否变化。"""
        payload = {
            "root": str(self.root),
            "repo_root": self.repo_root,
            "branch": self._git_branch(),
            "status": self._git_status(),
            "commits": self._git_recent_commits(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _git(self, args, fallback=""):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return result.stdout.strip() or fallback
        except Exception:
            return fallback

    def _git_branch(self) -> str:
        return self._git(["branch", "--show-current"], "-") or "-"

    def _git_status(self) -> str:
        return self._git(["status", "--short"], "clean") or "clean"

    def _git_recent_commits(self) -> str:
        lines = self._git(["log", "--oneline", "-5"]).splitlines()
        if not lines:
            return "- none"
        return "\n".join(f"- {line}" for line in lines)