import pandas as pd
from run import start_finatars

df = pd.read_csv("./evaluation/ragas_evaluation/final_evaluation_datasets/evaluation_dataset.csv")
df['web_data'] = None
df = df.astype({'answer_chunk_ids': 'object'})
df = df.astype({'web_data': 'object'})
agent_type = 'multi'  # 'single' or 'multi'
search_type = 'beam_search_advanced'  # if using 'single', select 'beam_search', 'beam_search_advanced' or 'vector_search'
web_search = True  # True if you want internet search, False otherwise.
# Start QA system
questions_list = list(df['question'])

for idx, q in enumerate(questions_list):
    answer, chunk_text, chunk_source, chunk_id, web_data = start_finatars(question=q, agent_type=agent_type,
                                                                          search_type=search_type,
                                                                          web_search=web_search,
                                                                          llm_config_path="./llm_config.json")
    # print(f'Answer:\n\n{answer}')
    # print(f'Information extracted from the web:\n\n{web_data}')
    if answer and len(answer) >= 3:
        df.loc[idx, 'answer'] = answer[2][1]
        df.at[idx, 'answer_chunk_ids'] = chunk_id
        df.at[idx, 'web_data'] = web_data




df.to_csv("./evaluation/ragas_evaluation/final_evaluation_datasets/evaluation_dataset_new_with_web_agent.csv")