from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import autogen
import re
import os
import json
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class MultiLLMSystem():
    """
    A class to manage a multi-language learning model (LLM) system, facilitating the configuration,
    interaction, and orchestration of user and assistant agents within a group chat environment.

    Attributes:
        llm_config (dict): The configuration for the LLM, loaded from a JSON file.
        llm_config_no_tools (dict): The LLM configuration excluding any tool functions.
        function_map (dict): A mapping of function names to their actual function calls.
        user_agent (UserProxyAgent): The user agent for the system.
        assistant_agents (list of AssistantAgents): A list of assistant agents added to the system.
    """

    def __init__(self, llm_config_path: str, retrieval_function):
        """
        Initializes the MultiLLMSystem with configurations and sets up the environment.

        :param llm_config_path: The path to the LLM configuration JSON file.
        """
        self.retrieval_function = retrieval_function
        llm_config = json.load(open(llm_config_path, 'r'))
        llm_config['config_list'][0]["api_key"] = OPENAI_API_KEY
        if self.retrieval_function.__name__ == 'search_graph':
            self.llm_config = llm_config.copy()
            self.llm_config['functions'] = [self.llm_config['functions'][0]]
        elif self.retrieval_function.__name__ == 'search_graph_and_web':
            self.llm_config = llm_config.copy()
            self.llm_config['functions'] = [self.llm_config['functions'][1]]
        self.llm_config_no_tools = {k: v for k, v in self.llm_config.items() if k != 'functions'}

    #### NO INTERNET SEARCH 
    def initialize_agents_no_web(self):
        #### USER AGENT
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            system_message='''You should start the workflow by first consulting the gatherer, then the reporter and finally the moderator. You have 
            to make sure that the Gatherer calls the `search_graph` function to 
            find relevant information, which is then given to the reporter and finally the moderator.
            The gatherer needs to use the `search_graph` function to find sufficient
            information, and if it is not invoked you must request that it does.
            '''
        )

        #### GATHERER
        self.gatherer = autogen.AssistantAgent(
            name="Gatherer",
            human_input_mode="NEVER",
            description='Agent that utilizes the search_graph function on the query to obtain data to pass to the next agent: reporter.',
            system_message=f'''
            As the Information Gatherer, you must start by using the `search_graph` function on the query and then follow these steps:

            1. Upon receiving a query, immediately invoke the `search_graph` function to obtain the results, which is the relevant documents and all of it's COMPLETE information as database that would serve as knowledge to the next agent: reporter.
            2. Present the ENTIRE retrieved documents to the Reporter, ensuring they have a comprehensive dataset to draft a response.
            3. Conclude your part to signal that you have finished collecting data and it is now ready for the Reporter to use in formulating the answer. 
            
            Remember, you are responsible for search document ONLY. The Reporter will rely on the accuracy and completeness of your findings to generate the final answer. So therefore, provide all of the data you have gathered without condensing any information.
            ''',
            max_consecutive_auto_reply=3,
            llm_config=self.llm_config
        )

        #### REPORTER
        self.reporter = autogen.AssistantAgent(
            name="Reporter",
            description='Agent that utilizes the data provided by the gatherer to summarize the information.',
            system_message='''
            As the Reporter, you are responsible for formulating an answer to the user's query using the information provided by the Gatherer. Here are the steps you need to follow:

            1. Wait for the Gatherer to complete their task and present you with a list of documents as data.
            2. Using the data provided by the Gatherer, these are the rules you need to follow to formulate the answer:
                2a. Analyze all the documents provided to you by the Gatherer, and understand the depth and knowledge provided in the data.
                2b. You are NOT allowed to add additional information outside of the data provided by the Gatherer, and rely SOLELY on that data to formulate your answer.
                2c. Understand FULLY each component of the query asked by the User, as well as the data provided by Gatherer to analyze what would be the best way to answer the users query from the data provided.
                2d. If the data provided does not fully answer the query, try your best to see if you can infer the answer from the data provided by the Gatherer.
            3. Finally, you need to follow the following rules to present and format your answer:
                3a. While formulating your answer, think carefully how a Financial Analyst would expect the answer to be. A financial analyst needs to have accurate, concise, relevant, and explainable information to be able to make their decisions. They are relying completely on your answer to make their judgement in their place of work. Hence, you need to tailor your response to their needs and style.
                3b. The answer MUST have only the information and not additional comments like "Based on documents/data/review...". It should be a well-formatted answer, with paragraphs, lists, bullet points if necessary.
                3c. Once you are done formulating your answer carefully, you pass your answer to the Moderator for a thorough review. Present your draft answer followed by "PLEASE REVIEW" for the Moderator to assess. You MUST type this at the end of your answer to indicate that it is for revising.
                3d. If the Moderator approves your answer, respond with "TERMINATE" to signal the end of the interaction. You must type "TERMINATE" at all costs to indicate the end of conversation.
                3e. If the data is insufficient to formulate your answer for the query, respond with an appropriate feedback about why it is insufficient, followed by "TERMINATE" to signal the end of interaction.
                3f. If the Moderator rejects your answer:
                    - Review their feedback.
                    - Make necessary amendments while still keeping all the above rules in mind.
                    - Resubmit the revised answer with "PLEASE REVIEW" at the end of your answer. You must type this without fail to indicate the review. And you must always submit the revised content to the Moderator in the same conversation, along with "PLEASE REVIEW". 
            4. You CANNOT disobey any rules, and need to abide by ALL rules. This is the law, and if you break the law, you are fined $1,000,000. So make sure you carefully follow the guidelines provided.
            ''',
            llm_config=self.llm_config_no_tools
        )

        #### MODERATOR
        self.moderator = autogen.AssistantAgent(
            name="Moderator",
            description='Agent that checks the formulated answer provided by the reporter and suggest changes if necessary.',
            system_message='''
            As the Moderator, your task is to review the Reporter\'s answers to ensure they meet the required criteria:
            - Assess the Reporter's answers after the "PLEASE REVIEW" prompt for alignment with the following criteria:
            A. Precision: Directly addressed the user's query with factual correct answer from given data.
            B. Depth: Provided comprehensive information using indexed content.
            C. Clarity: information presented logically and coherently.
            - Approve the answer by stating "APPROVED" if it meets the criteria. You must type "APPROVED" without fail to indicate that the answer has been approved.
            - If the answer falls short, specify which criteria were not met and instruct the Reporter to revise the answer accordingly, followed by "PLEASE REVISE" at the end of the answer. You must type "PLEASE REVISE" without fail to indicate that the answer has been asked for revision. 
            - Write your review in a structured format like paragraphs, lists, bullet points to showcase your review in a professional way.
            - Do not generate new content or answers yourself. If you generate an answer yourself, you are fined $1,000,000.
            Be critical in pointing out shortcomings. Your role is crucial in ensuring that the final answer provided to the user is factually correct and meets all specified quality standards.
            ''',
            llm_config=self.llm_config_no_tools
        )

        #### Register functions
        self.user_proxy.register_function(
            function_map={
                "search_graph": self.retrieval_function
            }
        )

        #### Define Conversation Flow
        def speaker_selection_function(last_speaker: autogen.Agent, groupchat: autogen.GroupChat):
            '''
            Define customized speaker selection function.
            '''
            messages = groupchat.messages

            if len(messages) <= 1:
                return self.gatherer

            if last_speaker is self.gatherer:
                # If Gatherer is calling search_graph function, continue
                if "function_call" in messages[-1].keys():
                    if messages[-1]["function_call"]["name"] == 'search_graph':
                        return "auto"
                # If Gatherer does not call search_graph, end the conversation
                else:
                    return None

            elif last_speaker is self.user_proxy:
                # If graph search complete, call reporter
                if "GRAPH SEARCH COMPLETE" in messages[-1]["content"]:
                    return self.reporter
                else:
                    return None

            elif last_speaker is self.reporter:
                # If reporter finishes summary, call moderator
                if "please review" in messages[-1]["content"].lower():
                    return self.moderator
                else:
                    return None

            elif last_speaker is self.moderator:
                # If moderator approves, end the conversation
                if "approved" in messages[-1]["content"].lower():
                    return None
                # If moderator asks to revise, call reporter
                elif "please revise" in messages[-1]["content"].lower():
                    return self.reporter
            else:
                return None

        #### DEFINE GROUPCHAT
        self.groupchat = autogen.GroupChat(
            agents=[self.user_proxy, self.gatherer, self.reporter, self.moderator],
            messages=[],
            max_round=20,
            speaker_selection_method=speaker_selection_function
        )

        #### DEFINE MANAGER
        self.manager = autogen.GroupChatManager(
            groupchat=self.groupchat,
            llm_config=self.llm_config_no_tools,
            system_message='''You should start the workflow in this order:
            1. gatherer, 
            2. reporter
            3. moderator. 
            If the gatherer does not use the "search_graph" function, 
            you MUST ABSOLUTELY request that it does.'''
        )

    #### WITH INTERNET SEARCH
    def initialize_agents_with_web(self):
        #### USER AGENT
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            system_message='''You should start the workflow by first consulting the gatherer, then the reporter and finally the moderator. You have 
            to make sure that the Gatherer calls the `search_graph_and_web` function to 
            find relevant information, which is then given to the reporter and finally the moderator.
            The gatherer needs to use the `search_graph_and_web` function to find sufficient
            information, and if it is not invoked you must request that it does.
            '''
        )

        #### GATHERER
        self.gatherer = autogen.AssistantAgent(
            name="Gatherer",
            human_input_mode="NEVER",
            description='Agent that utilizes the search_graph function on the query to obtain data to pass to the next agent: reporter.',
            system_message=f'''
            As the Information Gatherer, you must start by using the `search_graph_and_web` function on the query and then follow these steps:

            1. Upon receiving a query, immediately invoke the `search_graph_and_web` function to obtain the results, which is the relevant documents and all of it's COMPLETE information as database that would serve as knowledge to the next agent: reporter.
            2. Present the ENTIRE retrieved documents to the Reporter, ensuring they have a comprehensive dataset to draft a response.
            3. Conclude your part to signal that you have finished collecting data and it is now ready for the Reporter to use in formulating the answer. 
            
            Remember, you are responsible for search document ONLY. The Reporter will rely on the accuracy and completeness of your findings to generate the final answer. So therefore, provide all of the data you have gathered without condensing any information.
            ''',
            max_consecutive_auto_reply=3,
            llm_config=self.llm_config
        )

        #### REPORTER
        self.reporter = autogen.AssistantAgent(
            name="Reporter",
            description='Agent that utilizes the data provided by the gatherer to summarize the information.',
            system_message='''
            As the Reporter, you are responsible for formulating an answer to the user's query using the information provided by the Gatherer. Here are the steps you need to follow:

            1. Wait for the Gatherer to complete their task and present you with a list of documents as data.
            2. Using the data provided by the Gatherer, these are the rules you need to follow to formulate the answer:
                2a. Analyze all the documents provided to you by the Gatherer, and understand the depth and knowledge provided in the data.
                2b. You are NOT allowed to add additional information outside of the data provided by the Gatherer, and rely SOLELY on that data to formulate your answer.
                2c. Understand FULLY each component of the query asked by the User, as well as the data provided by Gatherer to analyze what would be the best way to answer the users query from the data provided.
                2d. If the data provided does not fully answer the query, try your best to see if you can infer the answer from the data provided by the Gatherer.
            3. Finally, you need to follow the following rules to present and format your answer:
                3a. While formulating your answer, think carefully how a Financial Analyst would expect the answer to be. A financial analyst needs to have accurate, concise, relevant, and explainable information to be able to make their decisions. They are relying completely on your answer to make their judgement in their place of work. Hence, you need to tailor your response to their needs and style.
                3b. The answer should not indicate words like "Based on information gathered", it should be a straight answer to the Analyst the way she would present/reporting it. Do not add additional comments suggesting that you are gathering information from any documents even if you are, just for clean answer presentation purposes.
                3c. Once you are done formulating your answer carefully, you pass your answer to the Moderator for a thorough review. Present your draft answer followed by "PLEASE REVIEW" for the Moderator to assess. You MUST type this at the end of your answer to indicate that it is for revising.
                3d. If the Moderator approves your answer, respond with "TERMINATE" to signal the end of the interaction. You must type "TERMINATE" at all costs to indicate the end of conversation.
                3e. If the data is insufficient to formulate your answer for the query, respond with an appropriate feedback about why it is insufficient, followed by "TERMINATE" to signal the end of interaction.
                3f. If the Moderator rejects your answer:
                    - Review their feedback.
                    - Make necessary amendments while still keeping all the above rules in mind.
                    - Resubmit the revised answer with "PLEASE REVIEW" at the end of your answer. You must type this without fail to indicate the review. 
            4. You CANNOT disobey any rules, and need to abide by ALL rules. This is the law, and if you break the law, you are fined $1,000,000. So make sure you carefully follow the guidelines provided.
            ''',
            llm_config=self.llm_config_no_tools
        )

        #### MODERATOR
        self.moderator = autogen.AssistantAgent(
            name="Moderator",
            description='Agent that checks the formulated answer provided by the reporter and suggest changes if necessary.',
            system_message='''
            As the Moderator, your task is to review the Reporter\'s answers to ensure they meet the required criteria:
            - Assess the Reporter's answers after the "PLEASE REVIEW" prompt for alignment with the following criteria:
            A. Precision: Directly addressed the user's query with factual correct answer from given data.
            B. Depth: Provided comprehensive information using indexed content. If there was more relevant information from gathered data that could have been added to the answer, please ask Reporter to revise.
            C. Clarity: information presented logically and coherently.
            - Approve the answer by stating "APPROVED" if it meets the criteria. You must type "APPROVED" without fail to indicate that the answer has been approved.
            - If the answer falls short, specify which criteria were not met and instruct the Reporter to revise the answer accordingly, followed by "PLEASE REVISE" at the end of the answer. You must type "PLEASE REVISE" without fail to indicate that the answer has been asked for revision. 
            - Write your review in a structured format like paragraphs, lists, bullet points to showcase your review in a professional way.
            - Do not generate new content or answers yourself. If you generate an answer yourself, you are fined $1,000,000.
            Be critical in pointing out shortcomings. Your role is crucial in ensuring that the final answer provided to the user is factually correct and meets all specified quality standards.
            ''',
            llm_config=self.llm_config_no_tools
        )

        #### Register functions
        self.user_proxy.register_function(
            function_map={
                "search_graph_and_web": self.retrieval_function
            }
        )

        #### Define Conversation Flow
        def speaker_selection_function(last_speaker: autogen.Agent, groupchat: autogen.GroupChat):
            '''
            Define customized speaker selection function.
            '''
            messages = groupchat.messages

            if len(messages) <= 1:
                return self.gatherer

            if last_speaker is self.gatherer:
                # If Gatherer is calling search_graph_and_web function, continue
                if "function_call" in messages[-1].keys():
                    if messages[-1]["function_call"]["name"] == 'search_graph_and_web':
                        return "auto"
                # If Gatherer does not call search_graph_and_web, end the conversation
                else:
                    return None

            elif last_speaker is self.user_proxy:
                # If graph and web search complete, call reporter
                if "GRAPH AND WEB SEARCH COMPLETE" in messages[-1]["content"]:
                    return self.reporter
                else:
                    return None

            elif last_speaker is self.reporter:
                # If reporter finishes summary, call moderator
                if "please review" in messages[-1]["content"].lower():
                    return self.moderator
                else:
                    return None

            elif last_speaker is self.moderator:
                # If moderator approves, end the conversation
                if "approved" in messages[-1]["content"].lower():
                    return None
                # If moderator asks to revise, call reporter
                elif "please revise" in messages[-1]["content"].lower():
                    return self.reporter
            else:
                return None

        #### DEFINE GROUPCHAT
        self.groupchat = autogen.GroupChat(
            agents=[self.user_proxy, self.gatherer, self.reporter, self.moderator],
            messages=[],
            max_round=20,
            speaker_selection_method=speaker_selection_function
        )

        #### DEFINE MANAGER
        self.manager = autogen.GroupChatManager(
            groupchat=self.groupchat,
            llm_config=self.llm_config_no_tools,
            system_message='''You should start the workflow in this order:
            1. gatherer, 
            2. reporter
            3. moderator. 
            If the gatherer does not use the "search_graph_and_web" function, 
            you MUST ABSOLUTELY request that it does.'''
        )

    def start_conversation(self, query):
        """
        Initiates the conversation in the group chat with a given query.

        :param query: The initial query or message to start the conversation.
        """
        self.user_proxy.initiate_chat(
            self.manager,
            message=query
        )

    def get_result(self):
        '''
        Function that returns the Multi-LLM Conversation as a List of tuples,
        Each tuple contains the name of the agent, followed by the content of the message.
        '''
        all_messages = self.groupchat.messages
        final_result = []
        for message in all_messages:
            if 'name' in message.keys():
                if message['name'] == 'user_proxy':
                    final_result.append(('User', message['content']))
                elif message['name'] == self.retrieval_function.__name__:
                    final_result.append(('Gatherer', message['content']))
                elif message['name'] == 'Reporter':
                    final_result.append(('Reporter', message['content']))
                elif message['name'] == 'Moderator':
                    final_result.append(('Moderator', message['content']))
        return final_result
