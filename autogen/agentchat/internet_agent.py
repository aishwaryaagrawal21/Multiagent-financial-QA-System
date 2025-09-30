from typing import Callable, Dict, Literal, Optional, Union

from .conversable_agent import ConversableAgent


class InternetAgent(ConversableAgent):
    """(In preview) Assistant agent, designed to solve a task with LLM.

    AssistantAgent is a subclass of ConversableAgent configured with a default system message.
    The default system message is designed to solve a task with LLM,
    including suggesting python code blocks and debugging.
    `human_input_mode` is default to "NEVER"
    and `code_execution_config` is default to False.
    This agent doesn't execute code by default, and expects the user to execute the code.
    """
    DEFAULT_SYSTEM_MESSAGE = """You are a smart internet assistant.
Your task is to extend the information provided by the DuckDuckGo API and present it in a group chat.
When you receive a snippet of information from the DuckDuckGo API, follow these steps:
    1. Analyze the snippet to understand the topic or question at hand.
    2. Use your language skills to expand on the topic, providing additional context, details, or explanations as needed.
    3. Format your response in a conversational tone suitable for a group chat environment.
    4. Ensure that your response is informative and adds value to the group discussion.
    5. If additional code execution is needed to fetch more information or perform a related task, provide that in a separate Python code block following the guidelines.
Remember to be concise, clear, and helpful. Finish the task efficiently and verify any facts before presenting them.
Reply "TERMINATE" when you have completed the task."""

    DEFAULT_DESCRIPTION = "An internet-savvy AI assistant that can pull information from the DuckDuckGo API and engage in group chats with informative and contextual responses."

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        description: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            name (str): agent name.
            system_message (str): system message for the ChatCompletion inference.
                Please override this attribute if you want to reprogram the agent.
            llm_config (dict): llm inference configuration.
                Please refer to [OpenAIWrapper.create](/docs/reference/oai/client#create)
                for available options.
            is_termination_msg (function): a function that takes a message in the form of a dictionary
                and returns a boolean value indicating if this received message is a termination message.
                The dict can contain the following keys: "content", "role", "name", "function_call".
            max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
                default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
                The limit only plays a role when human_input_mode is not "ALWAYS".
            **kwargs (dict): Please refer to other kwargs in
                [ConversableAgent](conversable_agent#__init__).
        """
        super().__init__(
            name,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            llm_config=llm_config,
            description=description,
            **kwargs,
        )

        # Update the provided description if None, and we are using the default system_message,
        # then use the default description.
        if description is None:
            if system_message == self.DEFAULT_SYSTEM_MESSAGE:
                self.description = self.DEFAULT_DESCRIPTION
