import csv
import argparse
import time
import logging
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.log_utils import setup_logging
from utils.env import init_seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_checkpoint', type=str, default="../../Translator/models/chatglm2-6b", required=False)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--distributed", action='store_const', default=False, const=True)
    parser.add_argument('--random_seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_workers", default=1, type=int)

    return parser.parse_args()


args = parse_args()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def read_arxiv_dataset():
    # paperid to node的映射和node to paperid的映射
    node2paperid = {}
    paperid2node = {}
    with open('../../data/arxiv/arxiv_nodeidx2paperid.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            nodeIdx = int(row[0])
            paperId = int(row[1])
            node2paperid[nodeIdx] = paperId
            paperid2node[paperId] = nodeIdx

    # 读取paperId到title和abstract映射的内容
    paperId2titleAndabs = pd.read_csv("../../data/arxiv/titleabs.tsv", delimiter='\t', header=None)
    paperId2titleAndabs = paperId2titleAndabs.rename(columns={0: "paper_id", 1: "title", 2: "abstract"})
    paperId2titleAndabs['node_id'] = paperId2titleAndabs['paper_id'].map(paperid2node).fillna(-1).astype(int)
    paperId2titleAndabs["title_abstract"] = "Title: " + paperId2titleAndabs["title"] + "\n" +"Abstract: " + paperId2titleAndabs["abstract"]
    paperId2titleAndabs = paperId2titleAndabs[paperId2titleAndabs['node_id'] != -1]

    paperId2titleAndabs = paperId2titleAndabs.replace('≤', '', regex=True)
    paperId2titleAndabs = paperId2titleAndabs.replace('≥', '', regex=True)
    paperId2titleAndabs = paperId2titleAndabs.replace('≠', '', regex=True)
    paperId2titleAndabs = paperId2titleAndabs.replace('≠', '', regex=True)
    paperId2titleAndabs = paperId2titleAndabs.replace('∫', '', regex=True)
    paperId2titleAndabs = paperId2titleAndabs.replace('∞', '', regex=True)
    paperId2titleAndabs = paperId2titleAndabs.replace('√', '', regex=True)

    sorted_paperId2titleAndabs = paperId2titleAndabs.sort_values(by='node_id')
    sample_neighbor_df = pd.read_csv("../../data/arxiv/sample_neighbor_df.csv")

    return sorted_paperId2titleAndabs, sample_neighbor_df


class LLM(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self._args = args
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self._args.llm_checkpoint, trust_remote_code=True)
        # model
        self.llm = AutoModel.from_pretrained(self._args.llm_checkpoint, trust_remote_code=True).half().to(device)

    def inference_chatglm_arxiv(self, arxiv_data, sample_neighbor_df):
        self.llm.eval()

        node_title_and_abs = arxiv_data.set_index('node_id')['title_abstract'].to_dict()
        src_to_dst_dict = sample_neighbor_df.groupby('src_node')['dst_node'].apply(list).to_dict()
        node2title = arxiv_data.set_index('node_id')['title'].to_dict()

        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} total paper count: {arxiv_data.shape[0]}")
        summary = []
        for data in arxiv_data.iterrows():
            node_id = data[1]['node_id']
            title = data[1]['title']
            src_prompt_pre = "The title and abstract of this paper are as follows: "
            src_prompt = '\n please summarize this paper and list five key words of this paper. All answers are in English and No Chinese in your answer'
            src_title_abstract = data[1]['title_abstract']
            node_word_input = src_prompt_pre + src_title_abstract
            if len(node_word_input[0]) > 3000- len(src_prompt):
                node_word_input = node_word_input[:3000-len(src_prompt)]
            node_word_input += src_prompt

            dst_prompt_pre = '\n The paper title and abstract are provided as follows: '
            dst_prompt = "\n Please summarize the topic and content of these papers. All answers are in English and No Chinese in your answer"
            dst_title_abstract = ""
            for neighbor_id in src_to_dst_dict[node_id]:
                dst_title_abstract = dst_title_abstract + node_title_and_abs[neighbor_id] + '\n'

            neighbor_word_input  = dst_prompt_pre + dst_title_abstract
            if len(neighbor_word_input[0]) > 3000-len(dst_prompt):
                neighbor_word_input = neighbor_word_input[:3000-len(dst_prompt)]
            neighbor_word_input += dst_prompt

            try:
                response_node, _ = self.llm.chat(self.tokenizer,
                                                        node_word_input ,
                                                        history=[])
                response_neighbor, _ = self.llm.chat(self.tokenizer,
                                                            neighbor_word_input,
                                                            history=[])
                summary.append({
                    'node_id': node_id,
                    'title': title,
                    'response_node': response_node,
                    'response_neighbor': response_neighbor
                })
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} paper {node_id+1} title: \"{title}\"")
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("CUDA out of memory error detected, skipping this batch")
                    continue
                else:
                    continue

        summary_df = pd.DataFrame(summary)
        embeddings = torch.load("../../data/arxiv/graphsage_node_embeddings.pt").to('cpu')
        new_data = []
        for _, row in summary_df.iterrows():
            node_id = int(row['node_id'])
            embedding = np.array(embeddings[node_id].detach())
            str_array = [str(num) for num in embedding]
            str_representation = ", ".join(str_array)
            title = node2title[row['node_id']]

            new_data.append({
                'node_id': node_id,
                'embedding':str_representation ,
                'paper_summary':row['response_node'],
                'citepapers_summary':row['response_neighbor'],
                'title':title
                })
        summary_embeddings = pd.DataFrame(new_data)
        summary_embeddings.to_csv('../../data/arxiv/summary_embeddings.csv',index=False)


def main():
    setup_logging()
    init_seeds(args.distributed, args.random_seed)

    logging.info("Main arguments:")
    for k, v in args.__dict__.items():
        logging.info("{}={}".format(k, v))
    logging.info("device type: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))

    # load model
    model = LLM(args)
    logging.info('start inference')
    arxiv_data, sample_neighbor_df = read_arxiv_dataset()
    model.inference_chatglm_arxiv(arxiv_data, sample_neighbor_df)


if __name__ == "__main__":
    main()
