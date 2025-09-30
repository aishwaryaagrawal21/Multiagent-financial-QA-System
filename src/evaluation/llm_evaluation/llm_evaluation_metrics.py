
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertForNextSentencePrediction
import torch
from rouge import Rouge

def calculate_perplexity(text, model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    encodings = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings['input_ids'])
        log_likelihood = outputs[0].item()
    perplexity = torch.exp(torch.tensor(log_likelihood / encodings['input_ids'].shape[1]))
    return perplexity.item()

def calculate_rouge_scores(hypothesis, reference):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return scores

def check_sentence_coherence(sentence_a, sentence_b, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForNextSentencePrediction.from_pretrained(model_name)
    encoded = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')
    with torch.no_grad():
        logits = model(**encoded)[0]
    coherence_score = torch.softmax(logits, dim=1)[0][0].item()
    return coherence_score

def evaluate_metrics(answers_csv, input_dataset_csv, output_metrics_csv):
    answers_df = pd.read_csv(answers_csv)
    input_df = pd.read_csv(input_dataset_csv)
    
    metrics_results = []
    for index, row in answers_df.iterrows():
        question = row['question']
        generated_answer = row['generated_answer']
        expected_answer = row['expected_answer']
        
        perplexity_gen = calculate_perplexity(generated_answer)
        perplexity_exp = calculate_perplexity(expected_answer)
        
        rouge_scores_gen = calculate_rouge_scores(generated_answer, expected_answer)
        rouge_scores_exp = calculate_rouge_scores(expected_answer, generated_answer)
        
        
        input_text = input_df[input_df['title'] == question]['sample_input_news'].iloc[0]
        coherence_score_gen = check_sentence_coherence(input_text, generated_answer)
        coherence_score_exp = check_sentence_coherence(input_text, expected_answer)
        
        metrics_results.append({
            'question': question,
            'generated_answer': generated_answer,
            'expected_answer': expected_answer,
            'perplexity_gen': perplexity_gen,
            'perplexity_exp': perplexity_exp,
            'rouge-1_gen': rouge_scores_gen['rouge-1']['f'],
            'rouge-2_gen': rouge_scores_gen['rouge-2']['f'],
            'rouge-l_gen': rouge_scores_gen['rouge-l']['f'],
            'coherence_score_gen': coherence_score_gen,
            'rouge-1_exp': rouge_scores_exp['rouge-1']['f'],
            'rouge-2_exp': rouge_scores_exp['rouge-2']['f'],
            'rouge-l_exp': rouge_scores_exp['rouge-l']['f'],
            'coherence_score_exp': coherence_score_exp
        })
    
    metrics_df = pd.DataFrame(metrics_results)
    metrics_df.to_csv(output_metrics_csv, index=False)
    print(f"Metrics saved to {output_metrics_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate metrics for generated answers and expected answers.')
    parser.add_argument('--answers_csv', type=str, required=True, help='CSV file with questions, generated answers, and expected answers. This should be the path to the file.')
    parser.add_argument('--input_dataset_csv', type=str, required=True, help='CSV file with the input dataset used for generating answers. This should be the path to the file.')
    parser.add_argument('--output_metrics_csv', type=str, default='output_metrics.csv', help='CSV file where the metrics results will be saved. This should be the path to the output file.')
    args = parser.parse_args()
    
    evaluate_metrics(args.answers_csv, args.input_dataset_csv, args.output_metrics_csv)
