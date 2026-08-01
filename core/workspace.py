import hashlib
import json
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class WorkspaceFingerprint:
    root_path: str
    repo_root: str
    branch: str
    status: str
    commits: str


class WorkspaceContext:
    def __init__(self, root: Path) -> None:
        self.root_path = Path(root).resolve()
        self._repo_root_path = self._find_repo_root(self.root_path)

    @classmethod
    def build(cls, cwd_path: str = ".") -> "WorkspaceContext":
        return cls(root=cwd_path)

    @staticmethod
    def _find_repo_root(start: Path) -> Path:
        current_path = start.resolve()
        for path in [current_path, *current_path.parents]:
            if (path / ".git").exists():
                return path
        return start

    @property
    def repo_root(self) -> str:
        return str(self._repo_root_path)

    def fingerprint(self) -> str:
        fingerprint = WorkspaceFingerprint(
            root_path=str(self.root_path),
            repo_root=self.repo_root,
            branch=self._git_branch(),
            status=self._git_status(),
            commits=self._git_recent_commits(),
        )
        return hashlib.sha256(
            json.dumps(asdict(fingerprint), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def text(self) -> str:
        """生成工作区摘要文本（用于 prompt 上下文）。"""
        branch = self._git_branch()
        status = self._git_status()
        commits = self._git_recent_commits()

        return textwrap.dedent(
            f"""\
            Workspace:
            - root: {self.root_path.name}
            - branch: {branch}
            - status:
            {status}
            - recent_commits:
            {commits}
            """
        ).strip()

    def _git(self, args, fallback=""):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.root_path,
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