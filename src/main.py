import transformers
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast, \
    T5Config, T5ForConditionalGeneration, T5TokenizerFast, ElectraConfig, ElectraTokenizerFast, ElectraForCausalLM
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from datasets import load_dataset
import sentencepiece
import time
import argparse
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        pass


def get_question(raw_question, insert):
    if "XXXXX" in raw_question:
        new_question = raw_question + "  " + str(insert)
    else:
        raise Exception("XXXXX not in question.")
    return new_question


def run_model_one_example(args, example, model, tokenizer, first_and_second):
    label_idx = example['options'].index(example['answer'])
    #print(example['answer'])
    first_sent = ' '.join(example['sentences']).replace("`",'')
    first_sents = [first_sent] * 10
    second_sent = example['question']
    second_sent_opt = [get_question(example['question'], option) for option in example['options']]
    second_sent_ans = [get_question(example['question'], example['answer'])] * 10
    first_second_sent_opt = [first_sents[i] + second_sent_opt[i] for i in range(10)]

    if first_and_second:
        sent_in = first_second_sent_opt
    else:
        sent_in = second_sent_opt

    if args.model_type == "d-o":
        tk_example_opt = tokenizer(sent_in, return_tensors="pt", padding=True).to(device)
    elif args.model_type == "e-o":
        tk_example_opt = tokenizer(sent_in, return_tensors="pt", truncation=True, padding=True).to(device)
    else:
        tk_example_opt = tokenizer(sent_in, return_tensors="pt", truncation=True, padding=True).to(device)
    output = model(**tk_example_opt, labels=tk_example_opt['input_ids'][label_idx].repeat(10, 1).to(device))

    tokenized = tk_example_opt ###
    ex_sm_logit_list = []
    last_tk_word_idx = None
    last_word_tks = None
    for i in range(10):
        word_ids = tokenized.word_ids(i) #which word token belongs to
        while last_word_tks is None:
            last_tk_word_idx = word_ids.pop()
            while last_tk_word_idx is None:
                last_tk_word_idx = word_ids.pop()
            last_word_tks = tokenized.word_to_tokens(last_tk_word_idx)
        last_word_tk_range = range(last_word_tks.start, last_word_tks.end)  #which tokens belong to last word
        last_word_ids = tokenized['input_ids'][i][last_word_tk_range]
        sm_logits = nn.Softmax(dim=1)(output.logits)
        last_word_sm_logit_list = []
        for j in range(len(last_word_ids)):
            last_word_sm_logit = sm_logits[i][last_word_tk_range[j]][last_word_ids[j]]
            last_word_sm_logit_list.append(last_word_sm_logit)
        avg_last_word_sm_logit = sum(last_word_sm_logit_list)/len(last_word_sm_logit_list)
        ex_sm_logit_list.append(avg_last_word_sm_logit.item())
    max_sm_logit = ex_sm_logit_list.index(max(ex_sm_logit_list))
    if label_idx == max_sm_logit:
        return 1
    else:
        return 0


def cbt_model_sparsity_experiment(args, number_examples, model, tokenizer, first_and_second):
    cbt = load_dataset("cbt", "CN", split="test") #"CN", "NE", "P", "V","raw"
    t0 = time.time()
    correct = 0
    for i in range(number_examples):
        correct += run_model_one_example(args, cbt[i], model, tokenizer, first_and_second)
    t1 = time.time() - t0
    correct_frac = correct/number_examples
    return correct, correct_frac, t1


def sparsity_experiment(args, exp_type, num_examples, large=False, sparsity_list=None, print_layers=False, first_and_second=True):
    if sparsity_list is None:
        sparsity_list = [0.10, 0.50, 0.90, 0.95, 0.99]
    results = []
    for sparse_lev in sparsity_list:
        if exp_type == "e-o":
            if large:
                model_type = "google/electra-large-generator"
            else:
                model_type = "google/electra-base-generator"
            tokenizer = ElectraTokenizerFast.from_pretrained(model_type)
            config = ElectraConfig.from_pretrained(model_type)
            model = ElectraForCausalLM.from_pretrained(model_type, config=config).to(device)
        elif exp_type == "d-o":
            if large:
                model_type = "gpt2-xl"
            else:
                model_type = "gpt2"
            model = GPT2LMHeadModel.from_pretrained(model_type).to(device)
            tokenizer = GPT2TokenizerFast.from_pretrained(model_type)
            tokenizer.pad_token = tokenizer.eos_token
        elif exp_type == "e-d":
            if large:
                model_type = "t5-3b"
            else:
                model_type = "t5-small"
            model = T5ForConditionalGeneration.from_pretrained(model_type).to(device)
            tokenizer = T5TokenizerFast.from_pretrained(model_type)
        else:
            raise Exception("Incorrect exp_type parameter. Choices are e-o, d-o, e-d.")

        print("Model type:", model_type)
        n_elements = []
        n_elements_zero = []
        for name, module in model.named_modules():
            #print(module)
            #print(type(module))
            if isinstance(module, transformers.pytorch_utils.Conv1D):
                prune.l1_unstructured(module, name='weight', amount=sparse_lev)
                n_elements_zero.append(float(torch.sum(module.weight == 0)))
                n_elements.append(float(module.weight.nelement()))
                if print_layers:
                    print("Sparsity in conv1.weight: {:.2f}%".format(100. * n_elements_zero[-1] /n_elements[-1]))
            if isinstance(module, nn.modules.linear.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparse_lev)
                n_elements_zero.append(float(torch.sum(module.weight == 0)))
                n_elements.append(float(module.weight.nelement()))
                if print_layers:
                    print("Sparsity in linear.weight: {:.2f}%".format(100. * n_elements_zero[-1] / n_elements[-1]))
        print("Total sparse: {:.2f}, Total trainable: {:.2f}, Global sparsity: {:.2f}%".format(sum(n_elements_zero), sum(n_elements),100. * sum(n_elements_zero) /sum(n_elements)))
        correct, correct_frac, t1 = cbt_model_sparsity_experiment(args, num_examples, model, tokenizer, first_and_second)
        results.append((model_type, sparse_lev, correct, correct_frac, time))
        print("Current results: " + str((model_type, sparse_lev, correct, correct_frac, t1)))
    pd.DataFrame(results).to_csv("data_" + exp_type + ".csv")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="d-o", choices=["d-o", "e-o", "e-d"])
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--first_and_sec", type=bool, default=False)
    parser.add_argument("--large", type=bool, default=False)
    args = parser.parse_args()

    sparsity_experiment(args, args.model_type, args.num_examples, large=args.large, sparsity_list=[0.0], first_and_second=args.first_and_sec)

