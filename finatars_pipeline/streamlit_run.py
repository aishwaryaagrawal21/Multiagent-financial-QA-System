# Imports
import streamlit as st
import os, io, subprocess, sys, re
from run import build_graph, start_finatars
import json
import asyncio

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Set up environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false" # For beam_search_advanced


# Function to load configurations
def load_config():
    with open('path_config.json', 'r') as config_file:
        return json.load(config_file)

# Use the loaded configuration
config = load_config()

# Initializing Parameters
rebuild_kg = False
agent_type = 'multi'
html_file_path = 'path_advanced.html'
triplet_path = config['triplet_data_path']
llm_config_path = config['llm_config_file']


# Initialize roles for finatars
ui_roles = {
    'User': {'name': 'Manager', 'icon': '👨‍💼', 'description': 'Recruits all agents'},
    'Gatherer': {'name': 'Data Gatherer', 'icon': '📊', 'description': 'Fetches relevant data from knowledge graph. Includes web data when web toggle is on.'},
    'Reporter': {'name': 'Analyst', 'icon': '🕵️‍♂️', 'description': 'Analyzes and reports the answer'},
    'Moderator': {'name': 'Critic', 'icon': '‍⚖️', 'description': 'Evaluates and approves content based on precision, depth and clarity. If content is not approved, it asks analyst or data gatherer to revise.'}
}

# Initialize sample questions
sample_questions = [
    "What could be the potential advantages and disadvantages of Nvidia and AMD in their product offerings in the chip industry?",
    "Microsoft plans to integrate Advanced AI into its Office 365 Suite, and VR into its XBOX gaming apps. How would this affect Samsung?",
    "What factors affect AMD in the AI sector?"
]

def display_chat_messages(entries):
    '''
    Function to display chat messages between different LLM roles
    Params:
    - entries (list of tuples): Each tuple consists of role, message

    Displays the messages as a conversation between different roles
    '''
    for persona, message in entries:
        ui_persona = ui_roles.get(persona, {'name': persona, 'icon': '🔊'})  # Default fallback
        st.markdown(f"""
        <div style="border-left: 3px solid #4CAF50; background-color: #f4f4f4; padding: 8px 16px; margin: 8px 0; border-radius: 8px;">
            <h4 style="color: #4CAF50;">{ui_persona['icon']} {ui_persona['name']}</h4>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)

# Function to display retrieved documents from graph
def display_retrieved_data(chunk_source, chunk_text):
    st.subheader('Retrieved Documents')
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Source**")
    with col2:
        st.markdown("**Text**")

    # Display data under each header
    for source, text in zip(chunk_source, chunk_text):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(source)
        with col2:
            st.write(text)

# Function to display data retrieved from the web
def display_web_data(web_data):
    st.subheader('Web Data')
    index=1
    for article in web_data:
        summary, url = article
        # Using the URL itself as the clickable text
        st.markdown(f"{index}. [{url}]({url})", unsafe_allow_html=True)
        st.write(summary)
        index+=1

# Build KG
build_graph(rebuild_kg = rebuild_kg, triplet_data_path = triplet_path)

st.set_page_config(layout="wide", page_title='Finatars')
st.title('Finatars: Answering Your Financial Queries')

# Sidebar for input question and navigation
with st.sidebar:
    go_to_overview = st.button('Overview')
    question = st.text_area('Enter your question here:', height=200, key='question_area',
                            help='Type your question and click Get Answer below')
    web_search = st.toggle('Include Web Data', value=False, help='Toggle to include web data in responses')
    get_answer = st.button('Get Answer')

    if go_to_overview:
        st.session_state['view'] = 'Overview'
    elif get_answer:
        with st.spinner('Finding the answer...'):
            answer, chunk_text, chunk_source, chunk_id, web_data = start_finatars(question=question,
                                                                                  agent_type=agent_type,
                                                                                  web_search=web_search)
            st.session_state['answer'] = answer
            st.session_state['chunk_text'] = chunk_text
            st.session_state['chunk_source'] = chunk_source
            if web_search:
                st.session_state['web_data'] = web_data
            st.session_state['view'] = 'Answer'

# Overview page
if 'view' not in st.session_state or st.session_state['view'] == "Overview":
    st.subheader("Sample Questions")
    for question in sample_questions:
        st.markdown(f"* {question}")

    st.subheader("Finatar roles")
    for role, details in ui_roles.items():
        st.markdown(f"{details['icon']} **{details['name']}**: {details['description']}", unsafe_allow_html=True)

# Sidebar radio buttons for navigation appear only after an answer is generated
if 'answer' in st.session_state:
    if web_search:
        view_options = ["Answer", "Retrieved Subgraph", "Retrieved Documents", "Web Data", "Internal LLM Conversation"]
    else:
        view_options = ["Answer", "Retrieved Subgraph", "Retrieved Documents", "Internal LLM Conversation"]
    selected_view = st.sidebar.radio("Viewing options:", options=view_options, index=0)
    st.session_state['view'] = selected_view

# Display content based on selected view
if 'view' in st.session_state and st.session_state['view'] != "Overview":
    if st.session_state['view'] == "Answer":
        st.subheader('Answer')
        reporter_answer = [a for a in st.session_state['answer'] if a[0] == 'Reporter'][-1] if st.session_state[
            'answer'] else ("Reporter", "No answer available")
        # Remove "PLEASE REVIEW" regardless of its case
        final_answer = re.sub(r"please review", "", reporter_answer[1], flags=re.IGNORECASE).strip()
        st.write(final_answer)
    elif st.session_state['view'] == "Retrieved Subgraph":
        st.subheader('Retrieved Subgraph')
        if os.path.exists(html_file_path):
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=480)
    elif st.session_state['view'] == "Retrieved Documents":
        display_retrieved_data(st.session_state['chunk_source'], st.session_state['chunk_text'])
    elif web_search and st.session_state['view'] == "Web Data":
        display_web_data(st.session_state['web_data'])
    elif st.session_state['view'] == "Internal LLM Conversation":
        st.subheader('Internal LLM Conversation')
        display_chat_messages(st.session_state['answer'])
