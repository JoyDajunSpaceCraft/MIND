import os
import json
import argparse
import torch
from tqdm import tqdm
from copy import copy
import logging
from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC


import os
from datetime import datetime

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("--use_memory", type=bool, default=False)
    parser.add_argument("--use_old", type=bool, default=False)
    parser.add_argument("--filter_method", type=str, default="cot",
                        choices=["no_filter", "conf", "cot", "conf_cot"])

    args = parser.parse_args()
    config_path = args.config_path
    use_memory = args.use_memory
    use_old = args.use_old
    filter_method = args.filter_method
    with open(config_path, "r") as f:
        args = json.load(f)

    args["use_memory"] = use_memory
    args["use_old"] = use_old
    args["filter_method"] = filter_method
    args = argparse.Namespace(**args)
    args.config_path = config_path
    if "shuffle" not in args:
        args.shuffle = False
    if "use_counter" not in args:
        args.use_counter = True
    
    return args


def main():
    args = get_args()
    logger.info(f" We have the args: {args}")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if args.use_old:
        current_time +="_use_old"
    else:
        current_time +="_use_new"
    current_time += "_" +args.filter_method
    output_folder_name = args.output_dir.split("/")[-1]
    
    # output dir
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    
    args.output_dir = os.path.join(args.output_dir, current_time)
    os.makedirs(args.output_dir)
    
    logger.info(f"output dir: {args.output_dir}")
    # save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    # create output file
    output_file = open(os.path.join(args.output_dir, "output.txt"), "w")

    # load data
    if args.dataset == "strategyqa":
        data = StrategyQA(args.data_path)
    elif args.dataset == "2wikimultihopqa":
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset == "hotpotqa":
        data = HotpotQA(args.data_path)
    elif args.dataset == "iirc":
        data = IIRC(args.data_path)
    else:
        raise NotImplementedError
    data.format(fewshot=args.fewshot)
    data = data.dataset
    if args.shuffle:
        data = data.shuffle()
    if args.sample != -1:
        samples = min(len(data), args.sample)
        data = data.select(range(samples))
    

    if args.use_old:
        from generate_old import DRAGIN as oldDRAGIN
        model = oldDRAGIN(args)
    else:
        from generate import DRAGIN
        model = DRAGIN(args)
    
    # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder_name = os.path.join("result", output_folder_name)
    logger_dir = os.path.join(output_folder_name, current_time)
    
    os.makedirs(logger_dir, exist_ok=True)
    # logger_dir = "results"
    for i in tqdm(range(len(data))):
        last_counter = copy(model.counter)        
        entry = data[i] 
        if args.use_old is False:
            model.set_logger(logger_dir, entry["qid"])
            model.logger.debug(f"Question: {entry['question']}")
        pred = model.inference(entry["question"], entry["demo"], entry["case"])
        pred = pred.strip()
        ret = {
            "qid": entry["qid"], 
            "prediction": pred,
        }
        if args.use_counter:
            ret.update(model.counter.calc(last_counter))
        output_file.write(json.dumps(ret)+"\n")
    

if __name__ == "__main__":
    # with torch.cuda.amp.autocast():
    with torch.no_grad():
        main()