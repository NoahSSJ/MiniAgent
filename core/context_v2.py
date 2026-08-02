from .session import Session
from .prompt import PromptManager












class ContextManager():
    def __init__(self, session: Session, prompt: PromptManager) -> None:
        self.session = session
        self.prompt = prompt

    

        