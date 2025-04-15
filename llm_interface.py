from abc import abstractmethod


class LLMInterface:
    def __init__(self):
        pass

    @abstractmethod
    def write(self):
        """
        Write a message to the language model.
        """
        pass
    
    @abstractmethod
    def read(self):
        """
        Read a message from the language model.
        """
        pass

    @abstractmethod
    def chat(self):
        """
        Write a single message and read the response from the language model.
        """
        pass