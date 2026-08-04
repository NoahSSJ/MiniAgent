from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import json
from typing import Optional, Any, Union
import uuid
from pathlib import Path

class Role(str, Enum):
    """Fuction Tools Role枚举类"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass
class Message():
    """消息类"""
    role: Role
    content: Optional[str] = None
    # 注意: 实际传的是 OpenAI 工具调用格式的 list, 如 agent.py 中的
    # [ {"id":..., "function":..., "type":"function"}, ... ]
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None


class Session():
    """Session类"""
    def __init__(self) -> None:
        self.session_id: str = str(uuid.uuid4())
        self.history: list[dict] = []
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()

    def add_message(self, role: Role, content: Optional[str] = None, tool_calls: Optional[list] = None, tool_call_id: Optional[str] = None) -> "Message":
        message = Message(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id
        )
        self.history.append(asdict(message))
        return message

    def get_history(self) -> list[dict]:
        return self.history.copy()
    
    def clear_history(self):
        self.history.clear()

    def to_dict(self):
        # datetime 需转成 ISO 字符串, 否则 json.dump 无法序列化(会抛 TypeError)
        return {
            "session_id": self.session_id,
            "history": self.history,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        session = cls()
        session.session_id = data['session_id']
        session.history = data['history']
        # ISO 字符串还原回 datetime, 保证 round-trip 后类型正确
        session.created_at = datetime.fromisoformat(data['created_at'])
        session.updated_at = datetime.fromisoformat(data['updated_at'])
        return session


class SessionManager:
    # save_dir: Path = Path(__file__).parent.parent / ".session"
    save_dir:Path = Path(r'D:\ai\MiniAgent\.session')

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.get_saved_sessions()

    def get_saved_sessions(self):
        sessions = [f for f in self.save_dir.glob("*.json") if f.is_file()]
        for session in sessions:
            with open(session, mode="r", encoding='utf-8') as f:
                data = json.load(f)
            session = Session.from_dict(data=data)
            self.sessions[session.session_id] = session

    def build(self) -> Session:
        session = Session()
        self.sessions[session.session_id] = session   # 持有对象，save 才有数据可用
        return session

    def save(self, session_id: str):
        if session_id not in self.sessions:
            raise KeyError(f"session_id not found: {session_id}")
        save_path = self.save_dir / f"{session_id}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.sessions[session_id].to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, session_id: str) -> Session:
        save_path = self.save_dir / f"{session_id}.json"
        if not save_path.is_file():                      # 修正：判断 is_file，而不是 is_dir
            raise FileNotFoundError(f"{save_path} not found")
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Session.from_dict(data)                   # 修正：用 Session 的类方法

    def delete(self, session_id: str):
        save_path = self.save_dir / f"{session_id}.json"
        save_path.unlink(missing_ok=True)
        self.sessions.pop(session_id, None)              # 修正：同步清理注册表

    def list_sessions(self):
        return list(self.sessions.keys())

        


# class Session():
#     save_dir: Path = Path(__file__).parent.parent / ".session"
#     def __init__(self) -> None:
#         self.session_id: str = str(uuid.uuid4())
#         self.history: list[dict] = []
#         self.created_at: datetime = datetime.now()
#         self.updated_at: datetime = datetime.now()
#         self.__post_init__()

#     @classmethod
#     def __post_init__(cls):
#         cls.save_dir.mkdir(parents=True, exist_ok=True)

#     def add_message(self, role: Role, content: Optional[str] = None, tool_calls: Optional[dict[str, Any]] = None, tool_call_id: Optional[str] = None) -> "Message":
#         message = Message(
#             role=role,
#             content=content,
#             tool_calls=tool_calls,
#             tool_call_id=tool_call_id
#         )
#         self.history.append(asdict(message))
#         return message

#     def get_history(self) -> list[dict]:
#         return self.history.copy()
    
#     def clear_history(self):
#         self.history.clear()

#     def to_dict(self):
#         return {
#             "session_id": self.session_id,
#             "history": self.history,
#             "created_at": self.created_at,
#             "updated_at": self.updated_at
#         }
    
#     @classmethod
#     def from_dict(cls, data: dict) -> "Session":
#         session = cls()
#         session.session_id = data['session_id']
#         session.history = data['history']
#         session.created_at = data['created_at']
#         session.updated_at = data['updated_at']
#         return session
    
#     def save(self):
#         save_path = self.save_dir / f"{self.session_id}.json"
#         with open(save_path, mode='w', encoding='utf-8') as f:
#             json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

#     @classmethod
#     def load(cls, session_id: str):
#         save_path = cls.save_dir / f"{session_id}.json"
#         if not save_path.is_dir():
#             return f"Error: {save_path} is not a dir"
#         if not save_path.is_file():
#             return f"Error: {save_path} is not a file"
#         with open(save_path, mode='r', encoding='utf-8') as f:
#             data = json.load(f)
#         session  = cls.from_dict(data=data)
#         return session
    
#     def delete(self, session_id: str):
#         save_path = self.save_dir / f"{session_id}.json"
#         if not save_path.is_dir():
#             return f"Error: {save_path} is not a dir"
#         if not save_path.is_file():
#             return f"Error: {save_path} is not a file"
#         save_path.unlink()
        
    






