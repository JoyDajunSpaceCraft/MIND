from typing import List, Tuple
from dataclasses import dataclass
import logging
import torch
import string
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput  # Output of .generate(...), useful for code hints
from transformers.generation.utils import GenerationMixin  # Use GenerationMixin.generate to check code
from transformers import PreTrainedModel, LlamaTokenizer  # Used for code hints
import logging
from retriever import BM25
from utils import RetrievalSystem, DocExtracter
# from SemanticIndexer import SemanticIndexer
import json
DEBUG = True
from spacy.pipeline import EntityRuler
from extract_relations import extract_relations_with_llama
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
nlp_dep = spacy.load("en_core_web_trf")  # Example use, if self.nlp is available, reuse it
nlp_dep.add_pipe("entityLinker", last=True)
ruler = EntityRuler(nlp_dep, overwrite_ents=True)

import spacy
from spacy.matcher import DependencyMatcher

@dataclass
class CheckerOutput:
    '''
    - hallucination: Whether hallucination is detected
    - curr_st: Start position of the hallucinated sentence
    - curr_en: End position of the hallucinated sentence
    - curr_thres: Whether each word in the hallucinated sentence exceeds the threshold
    '''
    hallucination: bool 
    curr_st: int = None
    curr_en: int = None
    curr_thres: List[bool] = None

# Initialize model
import torch
import torch.nn.functional as F
from math import log

@dataclass
class Block:
    text: str = None
    tokens: List[str] = None  # Storing tokens instead of ids makes merging words easier, especially handling "▁"
    range_: List[Tuple[int, int]] = None  # Each merged unit is considered a word. This keeps track of each word's span (left-closed, right-open)
    
    @property
    def len_tokens(self):
        return len(self.tokens)
    
    @property
    def len_words(self):
        return len(self.range_)

def merge_blocks(blocks: List[Block]) -> Block:
    text = "".join([block.text for block in blocks])
    tokens = sum([block.tokens for block in blocks], [])
    range_ = []
    st = 0
    for block in blocks:
        if block.range_:
            for l, r in block.range_:
                range_.append((st+l, st+r))
            st = range_[-1][1]
    return Block(text=text, tokens=tokens, range_=range_)

class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0
    
    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }
class Generator:
    def __init__(
        self,
        model_name_or_path: str
    ):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")  
        logger.info(f"Device = {self.model.device}")

        self.space_token = "▁"
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set padding token to end-of-sequence token

        # Define tokens that should not be merged
        self.tokens_cannot_merged = {
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode("0" + ch)[-1:])[0]
            # Adding "0" prevents merging with "▁"
            for ch in string.whitespace + string.punctuation
        } | {self.space_token, self.tokenizer.bos_token, self.tokenizer.eos_token}

    def simply_generate(
        self,
        input_text: str,
        max_length: int
    ) -> Tuple[bool, str]:
        """
        Generate text based on the input prompt.

        Returns:
            - ended (bool): Whether the generation has reached the end-of-sequence token.
            - new_text (str): The generated text.
        """
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)  
        input_length = input_ids.shape[1]

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            stop_strings="\n",
            tokenizer=self.tokenizer
        )[0, input_length:]

        if output_ids.shape[0] == 0:
            logger.info("Generated empty string in simply_generate()!")
            return True, ""

        if output_ids[0] == self.tokenizer.bos_token_id:
            output_ids = output_ids[1:]
        if output_ids[-1] == self.tokenizer.eos_token_id:
            return True, self.tokenizer.decode(output_ids[:-1])

        return False, self.tokenizer.decode(output_ids)

    def tokenize(
        self,        
        text: str,
        is_start: bool = False  # If False, remove the beginning-of-sequence token
    ):
        ids = self.tokenizer.encode(text)  # Convert text to token IDs
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        if not is_start and tokens[0] == self.tokenizer.bos_token:
            tokens = tokens[1:]
        return tokens
        
    def merge_tokens(
        self,
        tokens
    ) -> List[Tuple[int, int]]:
        """
        Merge token segments into word units.

        Returns:
            A list of tuples representing merged word ranges.
        """
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) \
                or tokens[i] in self.tokens_cannot_merged \
                or tokens[i-1] in self.tokens_cannot_merged:
                range_.append([i, i+1])  # Start a new word
            else:
                range_[-1][1] += 1
        return range_

    def build_block(
        self,        
        text: str,
        is_start: bool = False  # If False, remove the beginning-of-sequence token
    ) -> "Block":
        tokens = self.tokenize(text, is_start=is_start)
        range_ = self.merge_tokens(tokens)
        return Block(text=text, tokens=tokens, range_=range_)

    def generate(
        self,
        input_texts: List[str],  # Pre-segmented input: [demo, "\nQuestion:", question, "\nAnswer:", text]
        max_length: int,
    ) -> "GeneratorOutput":
        """
        Generate text and compute confidence signals based on entropy and attention.

        Returns:
            A `GeneratorOutput` object containing generated text, attention weights, entropy values, etc.
        """
        blocks = []
        for text in input_texts:
            blocks.append(self.build_block(text, is_start=not blocks))

        # 1) Merge question, text, etc. into input_ids
        input_tokens = sum([block.tokens for block in blocks], [])
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(input_tokens)], device=self.model.device)
        input_len_tokens = len(input_tokens)

        # 2) Generate new content (e.g., answer), stored in outputs.sequences
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            output_attentions=True,  # Optional: Retrieve attentions
            output_scores=True,  # Required for entropy calculation (only for newly generated tokens)
            stop_strings="\n",
            tokenizer=self.tokenizer
        )
        outputs: GenerateDecoderOnlyOutput

        # 3) Extract newly generated content
        all_ids = outputs.sequences[0]  
        new_token_ids = all_ids[input_len_tokens:]  
        tokens = self.tokenizer.convert_ids_to_tokens(new_token_ids)
        ended = (tokens[-1] == self.tokenizer.eos_token) if len(tokens) > 0 else False
        if ended: 
            tokens = tokens[:-1]
        text = self.tokenizer.convert_tokens_to_string(tokens)
        new_block = self.build_block(text, is_start=False)
        blocks.append(new_block)  

        # 4) Merge all blocks: now blocks include question, text, and newly generated content
        merged_blocks = merge_blocks(blocks)

        # 5) ============ Second Forward Pass ===============
        full_seq_ids = self.tokenizer.convert_tokens_to_ids(merged_blocks.tokens)
        full_seq_ids = torch.tensor([full_seq_ids], device=self.model.device)  # Shape: (1, total_seq_len)
        
        with torch.no_grad():
            full_out = self.model(
                full_seq_ids,
                output_attentions=True,    
                return_dict=True
            )

            attn = full_out.attentions[-1][0]  # Shape: (num_heads, total_seq_len, total_seq_len)
            attn = attn.mean(dim=0)  # Average across heads -> (total_seq_len, total_seq_len)
            
            # ---------- Token-to-Word Aggregation -----------
            # Convert token-level attention to word-level attention
            attn = torch.stack([
                attn[:, l:r].sum(dim=-1) for (l, r) in merged_blocks.range_
            ], dim=-1)  # Shape: (total_seq_len, merged_words)

            # Convert "attention giver" tokens to word-level attention
            final_atten = torch.stack([
                attn[l:r, :].mean(dim=0) for (l, r) in merged_blocks.range_
            ], dim=0)

            # Shape: (merged_words, merged_words)

            # ---------- Compute Maximum Attention -----------
            row_sum = final_atten.sum(dim=-1, keepdim=True) + 1e-10
            final_atten_norm = final_atten / row_sum
            max_atten, _ = final_atten_norm.max(dim=-1)  # Shape: (merged_words,)

            # ---------- Compute Entropy -----------
            all_logits = full_out.logits.squeeze(0)  # Shape: (total_seq_len, vocab_size)
            probs = torch.softmax(all_logits, dim=-1)  # Shape: (total_seq_len, vocab_size)
            token_entropy = (-probs * probs.log()).sum(dim=-1)  # Shape: (total_seq_len,)

            # Convert token-level entropy to word-level entropy
            word_entropy = torch.stack([
                token_entropy[l:r].sum() for (l, r) in merged_blocks.range_
            ], dim=0)  # Shape: (merged_words,)

        return GeneratorOutput(
            ended=ended,
            blocks=blocks,  
            merged_blocks=merged_blocks,  
            atten=final_atten,  
            max_atten=max_atten,  
            entropies=word_entropy,  
        )
@dataclass
class GeneratorOutput:
    '''
    Output values:
    - ended: Whether termination is detected (determined by eos_token)
    - blocks: Segmented storage of text blocks. Merging operations on words are done within blocks.
    - atten: (len_words, len_new_words). Already averaged across multiple heads. This refers to words after merging.
    - max_atten: (len_new_words,)
    - entropies: (len_new_words,)
    '''
    ended: bool
    blocks: List[Block] = None
    merged_blocks: Block = None
    atten: Tensor = None
    max_atten: Tensor = None
    entropies: Tensor = None
    mi: Tensor = None  
    
    @property
    def new_text(self):
        return self.blocks[-1].text
    
    @property
    def len_new_words(self):
        return self.blocks[-1].len_words

def compute_entity_confidences(outputs: GeneratorOutput, entities: List[dict], 
                               alpha=1.0, beta=0.2, agg_mode="max"):
    """
    Merge word-level entropy/attention values from outputs for each entity to form a single rank score.
    
    :param outputs: GeneratorOutput containing entropies & max_atten
    :param entities: A list of entity dictionaries, each containing:
                       - "start_word"
                       - "end_word"
                       - "text"
    :param alpha, beta: Hyperparameters for the aggregator
    :param agg_mode: str, one of ["max", "mean"] aggregator
    :return: A list of rank scores with the same length as `entities`
    """
    all_scores = []
    for ent in entities:
        st = ent.get("start_word", None)
        ed = ent.get("end_word", None)
        if st is None or ed is None:
            # If no alignment, fallback to 0 or skip
            all_scores.append(0.0)
            continue
        
        word_conf_values = []
        for w in range(st, ed):
            e_w = float(outputs.entropies[w].item())
            a_w = float(outputs.max_atten[w].item())
            conf_w = alpha * (1.0/(1.0 + e_w)) + beta * a_w
            word_conf_values.append(conf_w)
        
        if not word_conf_values:
            all_scores.append(0.0)
            continue
        
        if agg_mode == "max":
            score_e = max(word_conf_values)
        else:
            score_e = sum(word_conf_values)/len(word_conf_values)
        
        all_scores.append(score_e)
    return all_scores




class GeneratorWithMCDropout(Generator):
    def __init__(self, model_name_or_path: str, num_mc_samples=5):
        super().__init__(model_name_or_path)
        self.num_mc_samples = num_mc_samples
        
    def custom_generate(
        self,
        input_texts: List[str],  # [demo, "\nQuestion:", question, "\nAnswer:", text]
        max_length: int
    ) -> GeneratorOutput:
        """
        1) Use model.generate() to generate new text
        2) After completion, perform a second forward pass over merged_blocks (including question/text/new_block)
           to obtain global attention/entropy and aggregate token-to-word mappings.
        """
        # ---------- Step 1: Construct blocks (question + text) ----------
        blocks = []
        for text in input_texts:
            blocks.append(self.build_block(text, is_start=(len(blocks) == 0)))

        # Merge tokens into input_ids
        input_tokens = sum([block.tokens for block in blocks], [])
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(input_tokens)], device=self.model.device)
        input_len_tokens = len(input_tokens)

        # ---------- Step 2: Use model.generate() to create additional content (if any) ----------
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            return_dict_in_generate=True,
            output_attentions=False,  # This can be False since we'll perform a second forward pass
            output_scores=False,
            stop_strings="\n",
            tokenizer=self.tokenizer
        )

        # Extract newly generated content
        all_ids = outputs.sequences[0]  # shape: (input_len_tokens + gen_len,)
        new_token_ids = all_ids[input_len_tokens:]
        tokens_new = self.tokenizer.convert_ids_to_tokens(new_token_ids)
        ended = (len(tokens_new) > 0 and tokens_new[-1] == self.tokenizer.eos_token)
        if ended:
            tokens_new = tokens_new[:-1]  # Remove </s>

        text_new = self.tokenizer.convert_tokens_to_string(tokens_new)
        new_block = self.build_block(text_new, is_start=False)
        blocks.append(new_block)  # Append new generated content
        merged_blocks = merge_blocks(blocks)

        # ---------- Step 3: Perform second forward pass on the entire merged_blocks ----------
        full_seq_ids = self.tokenizer.convert_tokens_to_ids(merged_blocks.tokens)
        full_seq_ids = torch.tensor([full_seq_ids], device=self.model.device)  # shape(1, total_seq_len)

        with torch.no_grad():
            full_out = self.model(full_seq_ids, output_attentions=True, return_dict=True)
            attn = full_out.attentions[-1][0]  # (num_heads, total_seq_len, total_seq_len)
            attn = attn.mean(dim=0)  # (total_seq_len, total_seq_len)

            # Token-level to word-level mapping
            attn_summed = []
            for (l, r) in merged_blocks.range_:
                attn_summed.append(attn[:, l:r].sum(dim=-1)) 
            attn_summed = torch.stack(attn_summed, dim=-1)  # shape (total_seq_len, merged_words)

            final_atten_list = []
            for (l, r) in merged_blocks.range_:
                final_atten_list.append(attn_summed[l:r, :].mean(dim=0))
            final_atten = torch.stack(final_atten_list, dim=0)  # shape(merged_words, merged_words)

            # Normalize row-wise -> maxAtten
            row_sum = final_atten.sum(dim=-1, keepdim=True) + 1e-10
            final_atten_norm = final_atten / row_sum
            max_atten, _ = final_atten_norm.max(dim=-1)  # shape(merged_words,)

            # Compute token-level entropy -> word-level entropy
            logits = full_out.logits.squeeze(0)  # (total_seq_len, vocab_size)
            probs = torch.softmax(logits, dim=-1)  # (total_seq_len, vocab_size)
            token_entropy = (-probs * probs.log()).sum(dim=-1)  # (total_seq_len,)

            # Aggregate entropy at word level
            word_entropy = []
            for (l, r) in merged_blocks.range_:
                word_entropy.append(token_entropy[l:r].sum())
            word_entropy = torch.stack(word_entropy, dim=0)  # shape(merged_words,)

        # ---------- Step 4 (Optional): Compute Mutual Information (MI) ----------
        # If needed, MI can be computed via multiple forward passes in train mode
        word_mi = torch.zeros_like(word_entropy)  # Currently set to zero

        return GeneratorOutput(
            ended=ended,
            blocks=blocks,  # [question, text, new_block]
            merged_blocks=merged_blocks,  # Merged version of all blocks
            atten=final_atten,  # shape=(merged_words, merged_words)
            max_atten=max_atten,  # shape=(merged_words,)
            entropies=word_entropy,  # shape=(merged_words,)
            mi=word_mi  # shape=(merged_words,) (currently zero)
        )

def approximate_entity_span_in_newtext(outputs: GeneratorOutput, entity_text: str):
    """
    Try to find `entity_text` in `outputs.new_text` (the newly generated text).
    Return a (start_word, end_word) range if found, else (None, None).

    This is naive: if entity_text repeats, it just takes the first occurrence.
    If entity_text is not found, returns (None, None).
    """

    new_block = outputs.blocks[-1]  # the newly generated block
    text_all = new_block.text
    idx = text_all.find(entity_text)
    if idx == -1:
        return None, None

    # We found the substring at [idx, idx+len(entity_text)] in new_block.text
    start_char = idx
    end_char = idx + len(entity_text)

    # Now find which words in new_block.range_ overlap [start_char, end_char)
    # new_block.range_ is a list of (token_start_idx, token_end_idx) in token indices,
    # but we actually want character offsets. If not available, we'll guess.

    # In your code, new_block.range_ are word boundaries in "token indices", 
    # not character indices. So a direct match is tricky. 
    # We'll do a rough guess approach: 
    #   We'll convert tokens back to the string, 
    #   build a partial offset mapping, 
    #   then see which words overlap.

    # A simpler "back up" plan: we can split new_block.text by whitespace, 
    # and see which "word" we are in. But note new_block.text may not have
    # perfect whitespace. For demonstration:

    words = new_block.text.split()
    # We'll reconstruct them with a running char offset
    char_offset = 0
    start_word = None
    end_word = None

    current_word_idx = 0
    for w in words:
        wlen = len(w)
        # word covers [char_offset, char_offset + wlen)
        c_start = char_offset
        c_end = char_offset + wlen

        # if the entity substring overlaps this word
        if (c_end > start_char) and (c_start < end_char):
            if start_word is None:
                start_word = current_word_idx
            end_word = current_word_idx + 1  # end_word is exclusive
        char_offset += (wlen + 1)  # +1 for the whitespace we split on
        current_word_idx += 1

    # Now we have a word-level range in [start_word, end_word).
    # If we can't find an overlap, might return (None, None).
    return start_word, end_word

def join_if_nonempty(*li, sep=" "):
    return sep.join([s for s in li if len(s) > 0])

def match(word: str, real_words): 
    for real_word in real_words:
        if real_word in word:
            return True
    return False

def get_top_sentence(text):
    prev = ""
    for sent in nlp(text).sents:
        prev += sent.text
        sent = sent.text.strip()
        if len(sent) > 0:
            return prev
    return ""



class DRAGIN:
    def __init__(self, args):
        for k, v in args.__dict__.items():
            setattr(self, k, v)
        # Initialize the generator
        self.generator = GeneratorWithMCDropout(self.model_name_or_path)
        
        self.tokenizer = self.generator.tokenizer
        self.use_memory = args.use_memory

        # Initialize the retrieval system
        self.retriever = RetrievalSystem(
            retriever_name="BM25",  # Options: Contriever or BM25
            corpus_name="Wikipedia", 
            db_dir="/data_vault/pittnail/yuj49/rag_reason/SeaKR/corpus", 
            cache=False, 
            HNSW=False
        ) 
        self.counter = Counter()
        self.fact_memory = [] 
        self.max_memory_facts = 3  
    
    def filter_extracted_entities(self, extracted_data):
        """
        Filter extracted entities/relations based on `self.filter_method`.
        Returns a list of entity texts to be used for retrieval.
        
        Parameters:
          extracted_data: The result of `extract_relations_with_llama(...)`, formatted as:
                         {
                            "entities": [
                               {"text": "...", "type": "...", "rank_score": 0.XX},
                               ...
                            ],
                            "relations": [
                               {"subject":"...", "predicate":"...", "object":"...", "rank_score": 0.XX},
                               ...
                            ]
                         }
        """
        entities = extracted_data.get("entities", [])
        relations = extracted_data.get("relations", [])

        # Relations can also be used as query candidates,
        # or only entities can be used for retrieval, depending on the requirement.
        # Example: Concatenating (subject + predicate + object) into a query string.
        
        if self.filter_method == "no_filter":
            # 1) No filtering: return all entities
            return [e["text"] for e in entities]

        elif self.filter_method == "conf":
            # 2) Confidence-based filtering: select top-k entities based on `rank_score`
            top_k = 5  # Configurable based on experiments
            sorted_ents = sorted(entities, key=lambda x: x.get("rank_score", 0.0), reverse=True)
            top_ents = sorted_ents[:top_k]
            return [e["text"] for e in top_ents]
        
        elif self.filter_method == "cot":
            # 3) Chain-of-Thought (CoT) filtering: 
            #    This can use a predefined filtering logic to remove low-confidence or irrelevant entities.
            #    Example: Keep only entities with rank_score >= 0.5
            threshold = 0.5
            cot_ents = [e for e in entities if e.get("rank_score", 0) >= threshold]
            return [e["text"] for e in cot_ents]
        
        elif self.filter_method == "conf_cot":
            # 4) First apply CoT filtering, then sort by confidence
            threshold = 0.5
            filtered = [e for e in entities if e.get("rank_score", 0) >= threshold]
            # Then select the top-k based on `rank_score`
            top_k = 5
            sorted_ents = sorted(filtered, key=lambda x: x.get("rank_score", 0.0), reverse=True)
            top_ents = sorted_ents[:top_k]
            return [e["text"] for e in top_ents]
        
        else:
            # Default: return all entities
            return [e["text"] for e in entities]
    
    def set_logger(self, logger_dir, qid):
        """
        Set up logging.
        """
        if logger_dir is None:
            logger = logging.getLogger(__name__)
        else:
            logger = logging.getLogger(f"logger_{qid}")
            logger.setLevel(logging.DEBUG)
            import os
            handler = logging.FileHandler(os.path.join(logger_dir, f"{qid}.log"))
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
        self.logger = logger
        self.logger.setLevel(logging.DEBUG)
    
    def debug_print_blocks(self, outputs: GeneratorOutput):
        """
        Debug function to print block details.
        """
        print("=== Debug: Blocks Info ===")
        for i, block in enumerate(outputs.blocks):
            print(f"Block {i}:")
            print(f"  text = {repr(block.text)}")
            print(f"  len_text(chars) = {len(block.text)}")
            print(f"  len_tokens = {block.len_tokens}, len_words = {block.len_words}")
            # Uncomment below lines if detailed debugging is needed
            # print(f"  tokens = {block.tokens}")
            # print(f"  range_ = {block.range_}")
        print("=== End of Blocks Info ===\n")

    def render_top_facts(self, facts):
        """
        Format and return top retrieved facts.
        """
        if not facts:
            return "[No memory found]\n"
        lines = []
        for i, f in enumerate(facts):
            lines.append(f"{i+1}) {f['subject']} {f['relation']} {f['object']} (conf={f['confidence']:.2f})")
        return "\n".join(lines)

    def inference(self, question, demo, case):
        """
        Perform inference using the retrieval-augmented generation pipeline.
        """
        text = ""
        demo = "\n".join([d["case"] for d in demo])
        self.logger.debug("Starting inference process.")
        self.relations = None
        
        self.logger.debug("Initializing MemoryRetriever dynamically within inference(), facts=%s", self.relations)
        
        fail_count = 0
        while True:
            old_len = len(text)
            
            outputs = self.generator.custom_generate(
                input_texts=[demo, "\nQuestion:", question, "\nAnswer:", text],
                max_length=self.generate_max_length,
            )
            self.relations = extract_relations_with_llama(model=self.generator.model, 
                                                          tokenizer=self.generator.tokenizer, 
                                                          query=question)
            
            self.logger.debug("Extracted relations from question and generated text: %s", self.relations)
            self.logger.debug("Initial generated text: %s", outputs.new_text)

            if self.use_counter:
                self.counter.add_generate(outputs.new_text, self.generator.tokenizer)

            if outputs.new_text.strip() == "":
                self.logger.debug("Generated empty text, stopping inference.")
                break

            check_info = self.hallucination_check(outputs)
            if not check_info.hallucination:
                self.logger.debug("No hallucination detected.")
                text = join_if_nonempty(text, outputs.new_text.strip())
                self.logger.debug("Current generated text: %s", text)
                if outputs.ended or outputs.merged_blocks.len_tokens > self.generate_max_length:
                    self.logger.debug("Detected stop condition (EOS token or max length).")
                    break
            else:
                self.logger.debug("Hallucination detected. Proceeding with retrieval.")

                if not self.use_memory:
                    retrieve_qry = self.generate_retrieve_qry(outputs, check_info)
                    self.logger.debug("Retrieval query: %s", retrieve_qry)
                else:
                    retrieve_qry = self.generate_retrieve_qry_memory(outputs, check_info, self.relations)
                    self.logger.debug("Memory-enhanced retrieval query: %s", retrieve_qry)

                docs = self.retriever.retrieve(retrieve_qry, k=3, rrf_k=100)
                self.logger.debug("Retrieved supporting documents: %s", docs)
                
                prompt = demo + "\nContext:\n" + "\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(docs))
                prompt += "Answer in the same format as before.\n"
                prompt += outputs.blocks[1].text + outputs.blocks[2].text + outputs.blocks[3].text
                text = join_if_nonempty(text, self.generator.simply_generate(prompt, max_length=self.generate_max_length)[1].strip())

                if self.use_counter:
                    self.counter.hallucinated += 1

                self.logger.debug("Updated generated text: %s", text)

                if len(self.tokenizer.encode(text)) > self.generate_max_length:
                    self.logger.debug("Reached maximum text length. Stopping inference.")
                    break

            if old_len >= len(text):
                self.logger.info("Text length did not increase, stopping inference.")
                break

        self.logger.debug("Inference completed. Final generated text: %s", text)
        return text