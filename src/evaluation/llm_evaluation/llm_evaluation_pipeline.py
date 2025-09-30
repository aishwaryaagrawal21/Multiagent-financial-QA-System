import os, sys 
from    pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))



from util import get_embedding, run_query
from database_proxy import Neo4jGraphStore, VectorDBClient
from prompt import SINGLE_AGENT_QA_PROMPT, SINGLE_AGENT_QA_SYSTEM_PROMPT
from knowledge_graph import KnowledgeGraph
from beam_search_retrieval import Beam_Search_Retrieval
from QA import QAEngine
from typing_extensions import Annotated
from beam_search_retrieval import Beam_Search_Retrieval
from retrieval_functions import vector_retrieval
import numpy as np
from eval_neo4j.evaluation.llm_evaluation_metrics import calculate_perplexity, calculate_rouge_scores, check_sentence_coherence
import pandas as pd
import re
import argparse
from QA import QAEngine

evaluation_prompt = '''
Objective: Your task is to evaluate a generated answer based on its relevance and alignment with a specific input question, a set of relevant documents, and an ideal answer. Utilize the metrics of correctness, completeness, clarity, conciseness, and relevance for a comprehensive assessment. Each metric should be rated on a scale from 1 (indicating poor performance) to 5 (indicating excellent performance), with a particular emphasis on ensuring the generated answer's relevance to the input question and its avoidance of unnecessary details from the documents.

Evaluation Criteria:

1) Correctness (1-5): Measure the accuracy of the generated answer in reflecting the information found in the relevant documents and its consistency with the ideal answer. Highlight any inaccuracies or discrepancies noted.

2) Completeness (1-5): Evaluate whether the generated answer addresses all critical aspects and nuances of the topic as detailed in the relevant documents and the ideal answer. Identify any gaps or missing elements.

3) Clarity (1-5): Assess the generated answer for its ease of understanding and straightforwardness. Highlight areas of ambiguity or unnecessary complexity.

4) Conciseness (1-5): Review the generated answer for succinctness, ensuring it conveys essential information without extraneous content. Identify any unnecessary details or repetition.

5) Relevance (1-5): Ensure the generated answer directly addresses the input question without including irrelevant details from the documents. Score its relevance, noting instances where unrelated content is presented.

Evaluation Process:

Summary: Begin with a concise summary of the generated answer to set the foundation for your evaluation.
Scoring and Justification: For each metric, provide a score between 1 and 5, accompanied by a brief rationale. Utilize specific segments of the generated answer as examples to substantiate your evaluation.
Improvement Recommendations: Conclude with specific, actionable suggestions to improve the generated answer, with a focus on enhancing its relevance and addressing areas of concern identified in your ratings.
Output Format:

### Evaluation Summary:

- Offer a brief overview of the generated answer, noting initial observations relevant to your evaluation.

### Evaluation Details:

#### Correctness:
- Justification of the score based on alignment with documents and the ideal answer.

#### Completeness:
- Assessment of coverage regarding relevant topics.

#### Clarity:
- Evaluation of how understandable and direct the answer is.

#### Conciseness:
- Review of the answer's brevity and lack of unnecessary content.

#### Relevance:
- Analysis of the answer's focus on the input question and avoidance of irrelevant document details.

### Recommendations for Improvement:
- Provide targeted advice to enhance the generated answer, particularly its relevance to the input question.

### Final Scores:

1) **Correctness:** [Score] 
2) **Completeness:** [Score] 
3) **Clarity:** [Score] 
4) **Conciseness:** [Score]
5) **Relevance:** [Score] 

### Context:

**Input Question:** 
{}

**Relevant Documents:** 
{}

**Ideal Answer:** 
{}

**Generated Answer:** 
{}

'''

# for multi agent calling function
document_source = [] 
def search_graph(query:Annotated[str, "the question to ask"],
                 depth:Annotated[int, "the depth of the search"] =5, 
                 sim_threshold:Annotated[float, "threshold of the cosine similarity"] = 0.2,
                 limit = 3)->str:
    bs = Beam_Search_Retrieval(graph_client, vector_client, depth)
    doc_ids = bs.__search__(query, sim_threshold, limit)
    if not doc_ids:
        return "No document found"
    else:
        docs = vector_client.get_document_from_document_id(doc_ids)
        document, source = zip(*docs)
        # record the source 
        document_source.append("\n".join(document))
        return "\n".join(document)

def search_vector(query:Annotated[str, "the question to ask"],
                limit:Annotated[int, "the number of document to retrieve"] = 3, ):
    docs = vector_retrieval(vector_client, query, limit)
    #print(docs)
    vector_document, source = zip(*docs)
    return '\n'.join(vector_document)


##Retrieval function metrics
class Evaluation:
    def __init__(self, graph_client:Neo4jGraphStore, vector_client:VectorDBClient, config_path='', search = 'beam_search', is_eval = True):
        #evaluation options: 'graph', 'retrieval', 'conversation',
        self.graph_client = graph_client
        self.vector_client = vector_client
        self.extraction_prompt_template = SINGLE_AGENT_QA_PROMPT
        self.single_agent_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.is_eval = is_eval
        self.single_agent_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.qa_engine = QAEngine(graph_client, vector_client)
        self.config_path = config_path

    def get_documents(self, question, depth, sim_threshold, limit = 5, search='beam_search'):
        if search=='beam_search':
            bs = Beam_Search_Retrieval(self.graph_client, self.vector_client, depth)
            doc_ids = bs.__search__(question, sim_threshold)
            docs = self.vector_client.get_document_from_document_id(doc_ids)
            #print(docs)
            beam_document, source = zip(*docs)
            return '\n'.join(beam_document)
        elif search == 'vector_search':
            docs = vector_retrieval(self.vector_client, question, limit)
            #print(docs)
            vector_document, source = zip(*docs)
            return '\n'.join(vector_document)
        else:
            bs = Beam_Search_Retrieval(self.graph_client, self.vector_client, depth)
            doc_ids = bs.__search__(question, sim_threshold)
            docs = self.vector_client.get_document_from_document_id(doc_ids)
            #print(docs)
            beam_document, source = zip(*docs)

            docs = vector_retrieval(self.vector_client, question, limit)
            #print(docs)
            vector_document, source = zip(*docs)

            return '\n'.join(beam_document), '\n'.join(vector_document)


    def calculate_embedding_similarity(self, emb1, emb2):
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        similarity = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
        return similarity

    def document_retrieval_eval(self, question, depth, sim_threshold, limit = 3):
        res1 = {}
        res2 = {}
        res1['method'] = 'Beam Retrieval'
        res2['method'] = 'Vector Retrieval'

        qs_emb = get_embedding(question)

        doc1, doc2 = self.get_documents(question, depth, sim_threshold, limit = limit)

        doc1_emb = get_embedding(doc1)
        doc2_emb = get_embedding(doc2)

        cosine_score1 = self.calculate_embedding_similarity(doc1_emb, qs_emb)
        cosine_score2 = self.calculate_embedding_similarity(doc2_emb, qs_emb)

        print(f'Cosine similarity:\nBeam Retrieval: {cosine_score1}\nVector Retrieval: {cosine_score2}')

    def get_retrieval_similarity(self, question, doc):
        qs_emb = get_embedding(question)
        doc1_emb = get_embedding(doc)
        cosine_score1 = self.calculate_embedding_similarity(doc1_emb, qs_emb)
        return cosine_score1

    def QA_eval(self, question, ideal_answer, depth=3, sim_threshold=0.3, limit = 3):
        answers = {}
        source = {}
        evals = {}
        evals['question'] = question 
        evals['ideal_answer'] = ideal_answer
        answers['beam_search_sa'], source['beam_search_sa'] = self.qa_engine.single_agent_qa(question,sim_threshold=0.2, retrieval_method = 'beam_search')
        answers['vector_search_sa'], source['beam_search_sa'] = self.qa_engine.single_agent_qa(question,sim_threshold=0.2, retrieval_method = 'vector_search')
        #answers['beam_search_ma']


        for k, v in answers.items():
            fin_prompt = evaluation_prompt.format(question, source[k], v, ideal_answer)
            eval_ = run_query(fin_prompt, SINGLE_AGENT_QA_SYSTEM_PROMPT, self.single_agent_model)
            evals[k+'_answer'] = answers[k]
            evals[k] = eval_ 
        
        return evals 

    def get_LLM_metrics(self, sample_text):
        final_scores = {}
        try:
            scores = re.findall(r"\*\*Correctness:\*\* (\d+(\.\d+)?)", sample_text)
            final_scores['Correctness'] = scores[0][0]
            scores = re.findall(r"\*\*Completeness:\*\* (\d+(\.\d+)?)", sample_text)
            final_scores['Completeness'] = scores[0][0]
            scores = re.findall(r"\*\*Clarity:\*\* (\d+(\.\d+)?)", sample_text)
            final_scores['Clarity'] = scores[0][0]
            scores = re.findall(r"\*\*Conciseness:\*\* (\d+(\.\d+)?)", sample_text)
            final_scores['Conciseness'] = scores[0][0]
            scores = re.findall(r"\*\*Relevance:\*\* (\d+(\.\d+)?)", sample_text)
            final_scores['Relevance'] = scores[0][0]
        except:
            pass

        print(final_scores)
        return final_scores

    def run_LLM_eval(self, question, source, generated_answer, ideal_answer):
        fin_prompt = evaluation_prompt.format(question, source, generated_answer, ideal_answer)
        eval_ = run_query(fin_prompt, SINGLE_AGENT_QA_SYSTEM_PROMPT, self.single_agent_model)
        return eval_

    def generate_conversation_metrics(self, answers_csv, input_dataset_csv, output_metrics_csv, method = 'beam_search'):
        answers_df = pd.read_csv(answers_csv)
        input_df = pd.read_csv(input_dataset_csv)
        
        metrics_results = []
        
        for index, row in answers_df.iterrows():
            question = row['question']
            expected_answer = row['baseAnswer']
            generated_answer, sources = self.qa_engine.single_agent_qa(question,sim_threshold=0.2, retrieval_method =method)
            retrieval_metric = self.get_retrieval_similarity(question, ''.join(sources))
            llm_overall_eval = self.run_LLM_eval(question, sources, generated_answer, expected_answer )
            #print(llm_overall_eval)
            perplexity_gen = calculate_perplexity(generated_answer)
            perplexity_exp = calculate_perplexity(expected_answer)
            
            rouge_scores_gen = calculate_rouge_scores(generated_answer, expected_answer)
            rouge_scores_exp = calculate_rouge_scores(expected_answer, generated_answer)
            
            
            '''input_text = input_df[input_df['title'] == question]['sample_input_news'].iloc[0]
            coherence_score_gen = check_sentence_coherence(input_text, generated_answer)
            coherence_score_exp = check_sentence_coherence(input_text, expected_answer)'''
            
            metrics_results.append({
                'question': question,
                'generated_answer': generated_answer,
                'expected_answer': expected_answer,
                'llm_eval': llm_overall_eval,
                'retrieval_metric': retrieval_metric,
                'perplexity_gen': perplexity_gen,
                'perplexity_exp': perplexity_exp,
                'rouge-1_gen': rouge_scores_gen['rouge-1']['f'],
                'rouge-2_gen': rouge_scores_gen['rouge-2']['f'],
                'rouge-l_gen': rouge_scores_gen['rouge-l']['f'],
                #'coherence_score_gen': coherence_score_gen,
                'rouge-1_exp': rouge_scores_exp['rouge-1']['f'],
                'rouge-2_exp': rouge_scores_exp['rouge-2']['f'],
                'rouge-l_exp': rouge_scores_exp['rouge-l']['f'],
                #'coherence_score_exp': coherence_score_exp,
            })
            #break
    
        metrics_df = pd.DataFrame(metrics_results)
        metrics_df.to_csv(output_metrics_csv, index=False)
        print(f"Metrics saved to {output_metrics_csv}")

    def m_generate_conversation_metrics(self, answers_csv, input_dataset_csv, output_metrics_csv, config_path, method = 'beam_search', agent_type = 'single_agent'):
        '''
        multi agent generation
        '''
        answers_df = pd.read_csv(answers_csv)
        input_df = pd.read_csv(input_dataset_csv)
        retrievel_function = search_graph if method == "beam_search" else search_vector 
        metrics_results = []
        llm_config_path = self.config_path #"/Users/krishabhambani/Downloads/finatars/eval_neo4j/llm_config.json"
        for index, row in answers_df.iterrows():
            try:
                question = row['question']
                expected_answer = row['baseAnswer']
                generated_answer = self.qa_engine.multi_agent_qa(question, retrievel_function, llm_config_path)
                sources = self.get_documents(question, 3, sim_threshold=0.3, limit = 3, search=method) #"Nothing" if not document_source else document_source.pop()

                retrieval_metric = self.get_retrieval_similarity(question, ''.join(sources))
                llm_overall_eval = self.run_LLM_eval(question, sources, generated_answer, expected_answer )
                
                final_metrics = self.get_LLM_metrics(llm_overall_eval)
                #print(llm_overall_eval)
                perplexity_gen = calculate_perplexity(generated_answer)
                perplexity_exp = calculate_perplexity(expected_answer)
                
                rouge_scores_gen = calculate_rouge_scores(generated_answer, expected_answer)
                rouge_scores_exp = calculate_rouge_scores(expected_answer, generated_answer)
                

                
                '''input_text = input_df[input_df['title'] == question]['sample_input_news'].iloc[0]
                coherence_score_gen = check_sentence_coherence(input_text, generated_answer)
                coherence_score_exp = check_sentence_coherence(input_text, expected_answer)'''
                
                metrics_results.append({
                    'question': question,
                    'generated_answer': generated_answer,
                    'expected_answer': expected_answer,
                    'retrieval_metric': retrieval_metric,
                    'llm_eval': llm_overall_eval,
                    'Correctness': final_metrics['Correctness'],
                    'Completeness': final_metrics['Completeness'],
                    'Clarity': final_metrics['Clarity'],
                    'Conciseness': final_metrics['Conciseness'],
                    'Relevance': final_metrics['Relevance'],
                    'perplexity_gen': perplexity_gen,
                    'perplexity_exp': perplexity_exp,
                    'rouge-1_gen': rouge_scores_gen['rouge-1']['f'],
                    'rouge-2_gen': rouge_scores_gen['rouge-2']['f'],
                    'rouge-l_gen': rouge_scores_gen['rouge-l']['f'],
                    #'coherence_score_gen': coherence_score_gen,
                    'rouge-1_exp': rouge_scores_exp['rouge-1']['f'],
                    'rouge-2_exp': rouge_scores_exp['rouge-2']['f'],
                    'rouge-l_exp': rouge_scores_exp['rouge-l']['f'],
                    #'coherence_score_exp': coherence_score_exp,
                })
            except:
                break
    
        metrics_df = pd.DataFrame(metrics_results)
        metrics_df.to_csv(output_metrics_csv, index=False)
        print(f"Metrics saved to {output_metrics_csv}")


    def evaluate_conversation(self, answers_csv, input_dataset_csv, output_metrics_csv='output.csv'):
        # method = 'beam_search'
        # agent_setup = 'single_agent'
        # print('********', method, agent_setup)
        # self.generate_conversation_metrics(answers_csv, input_dataset_csv, 's_'+method+'_'+output_metrics_csv, method)
        # method = 'vector_search'
        # agent_setup = 'single_agent'
        # print('********',method, agent_setup)
        # self.generate_conversation_metrics(answers_csv, input_dataset_csv, 's_'+method+'_'+output_metrics_csv, method)
        method = 'beam_search'
        agent_setup = 'multi_agent'
        print('********',method, agent_setup)
        self.m_generate_conversation_metrics(answers_csv, input_dataset_csv, "m_"+ method+'_'+output_metrics_csv, method)
        method = 'vector_search'
        agent_setup = 'multi_agent'
        print('********',method, agent_setup)
        self.m_generate_conversation_metrics(answers_csv, input_dataset_csv, "m_"+ method+'_'+output_metrics_csv, method)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add finatars folder absolute path')
    parser.add_argument('--finatars_path')
    args = parser.parse_args()

    # connect to database 
    username = "neo4j"
    password = "testgraph"
    url = "bolt://localhost:7687"
    database = "neo4j"

    graph_client = Neo4jGraphStore(username=username, password=password,url=url,database=database)
    vector_client = VectorDBClient("localhost")
    config_path = os.path.join(args.finatars_path, "eval_neo4j/llm_config.json")

    eval_obj = Evaluation(graph_client, vector_client, config_path)
    #question = "What are the 5 important factor that affects Microsoft?"
    #/Users/krishabhambani/Downloads/finatars/
    #eval_obj.document_retrieval_eval(question, depth=3, sim_threshold=0.2, limit = 3)
    answers_csv =os.path.join(args.finatars_path, "answer_eval_metrics/Questions_Answering_data_for_Eval.csv")
    input_dataset_csv = os.path.join(args.finatars_path, "data_collection/sample_input_news.csv")
    eval_obj.evaluate_conversation(answers_csv,input_dataset_csv)

        

