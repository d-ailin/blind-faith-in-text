from abc import ABC, abstractmethod

from PIL import Image


class BaseAdapter(ABC):
    def __init__(
        self, 
        model=None, 
        tokenizer=None, 
    ):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def generate(
        self,
        query: str,
        image: Image,
        image_path: str,
        **kwargs
    ) -> dict:
        pass
    
    @abstractmethod
    def raw_generate(self, messages: list, **kwargs) -> dict:
        pass
    
    @abstractmethod
    def construct_messages(self, messages: list) -> dict:
        pass

    def __call__(
        self,
        query: str,
        image: str,
        image_path: str,
        **kwargs
    ) -> dict:
        return self.generate(query, image=image, image_path=image_path, **kwargs)