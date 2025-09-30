import streamlit as st
import pandas as pd
import ast

# Assuming df is your DataFrame with columns chunk_id, question, and answer
df = pd.read_csv('data/qa_data_0.csv')

# Function to display chunk_id, question, and answer in a big window pane
def display_chunk(chunk_id, question, answer):

    #st.write(f"## Chunk ID: {chunk_id}")   
    question = ast.literal_eval(question)
    answer = ast.literal_eval(answer)
    i = st.selectbox('Select article number:', list(range(len(question))))

    #for i in range(len(question)):
    st.write("Article ", i)
    st.write(f"**Chunk sources:** {question[i]}")
    st.write(f"**Chunk Text:** {answer[i]}")

# Streamlit app
st.title('Chunk Viewer')

# Select a chunk by chunk_id
selected_chunk_id = st.selectbox('Select Chunk ID:', df.index)

# Display the selected chunk
selected_chunk = df.iloc[selected_chunk_id]
display_chunk(selected_chunk['chunk_id'], selected_chunk['chunk_source'], selected_chunk['chunk_text'])

# Editable text boxes for question and answer
edited_question = st.text_area('Edit Question:', selected_chunk['question'])
edited_answer = st.text_area('Edit Answer:', selected_chunk['answer'])

# Save button
if st.button('Save Changes'):
    # Update the DataFrame with edited question and answer
    df.loc[selected_chunk_id, 'question'] = edited_question
    df.loc[selected_chunk_id, 'answer'] = edited_answer
    st.success('Changes saved successfully!')
    # Optional: Display the updated DataFrame
    st.write('## Updated DataFrame')
    st.write(df)
    df.to_csv('data/qa_data_0.csv')


