import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_json_from_response(response: str):
    """
    从模型生成的响应中提取实体和关系信息，并拼凑成 JSON 格式。
    """
    try:
        entities = []
        relations = []

        # 定义模式匹配实体和关系
        entity_pattern = r'"text":\s*"([^"]+)",\s*"type":\s*"([^"]+)"'
        relation_pattern = r'"subject":\s*"([^"]+)",\s*"predicate":\s*"([^"]+)",\s*"object":\s*"([^"]+)"'

        # 按顺序提取实体信息
        for entity_match in re.findall(entity_pattern, response):
            entities.append({"text": entity_match[0], "type": entity_match[1]})

        # 按顺序提取关系信息
        for relation_match in re.findall(relation_pattern, response):
            relations.append({
                "subject": relation_match[0],
                "predicate": relation_match[1],
                "object": relation_match[2]
            })

        # 如果提取到的内容不为空，则拼凑为 JSON
        if entities or relations:
            return {"entities": entities, "relations": relations}

        # 如果没有提取到有效内容，返回空结构
        # print("No valid data extracted from response.")
        return {"entities": [], "relations": []}

    except Exception as e:
        # print(f"Error during JSON extraction: {e}")
        return {"entities": [], "relations": []}


def extract_relations_with_llama(model, tokenizer, query: str, text: str = None, include_context: bool = True) -> dict:
    """
    使用 Llama 提取实体和关系。
    根据 `include_context` 决定是否使用 `context`。
    """
    if include_context:
        # Prompt 包含 context 和 query
        prompt = f"""
        Below is a context and a query. Extract all entities (including PERSON, EVENT, WORK_OF_ART, etc.) and relationships (e.g., mother_of, directed_by, etc.) from the context and query in a structured format.
        You should only generate based on the text in the input query and context don't generate extra text.
        Example:
        Query: Who is the mother of the director of the film Polish-Russian War?
        Context: The director of Polish-Russian War is John Smith, whose mother is Maria.
        Expected output format (JSON):
        {{
            "entities": [
                {{"text": "John Smith", "type": "PERSON"}},
                {{"text": "Maria", "type": "PERSON"}},
                {{"text": "Polish-Russian War", "type": "EVENT"}}
            ],
            "relations": [
                {{"subject": "John Smith", "predicate": "mother_of", "object": "Maria"}},
                {{"subject": "John Smith", "predicate": "directed_by", "object": "Polish-Russian War"}}
            ]
        }}
        Query: {query}
        Context: {text}

        Expected output format (JSON):
        """
    else:
        # Prompt 仅包含 query
        prompt = f"""
        Below is a query. Extract all entities (including PERSON, EVENT, WORK_OF_ART, etc.) and relationships (e.g., mother_of, directed_by, etc.) from the query in a structured format.
        You should only generate based on the text in the input query don't generate extra text.
        Example:
        Query: Who is the mother of the director of the film Polish-Russian War?
        Expected output format (JSON):
        {{
            "entities": [
                {{"text": "Polish-Russian War", "type": "EVENT"}}
            ],
            "relations": [
                {{"subject": "Who", "predicate": "directed_by", "object": "Polish-Russian War"}}
            ]
        }}
        Query: {query}

        Expected output format (JSON):
        """

    # 模型生成
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print("response", response[len(prompt):])

    # 提取和解析生成的 JSON
    structured_data = extract_json_from_response(response[len(prompt):])
    if structured_data != {'entities': [], 'relations': []}:
        print("Extracted structured data:")
        return structured_data
    else:
        return {"entities": [], "relations": []}
# def extract_relations_with_llama(model, tokenizer, query: str, text: str = None, include_context: bool = True) -> dict:
#     """
#     让 LLM 通过 Chain-of-Thought (CoT) 进行 reasoning 提取实体和关系，并生成 rank score。
#     """
#     if include_context:
#         # 使用 CoT 方法让 LLM 进行 reasoning
#         prompt = f"""
#         Below is a context and a query. Extract key entities and relationships using step-by-step reasoning.
#         Rank the extracted entities based on their relevance to the query.

#         Example:
#         Query: Who is the mother of the director of the film Polish-Russian War?
#         Context: The director of Polish-Russian War is John Smith, whose mother is Maria.
#         Reasoning:
#         - Identify "Polish-Russian War" as an important event.
#         - Find "John Smith" as the director.
#         - Recognize "Maria" as the mother of John Smith.
#         Ranked output (JSON):
#         {{
#             "entities": [
#                 {{"text": "John Smith", "type": "PERSON", "rank_score": 0.95}},
#                 {{"text": "Maria", "type": "PERSON", "rank_score": 0.90}},
#                 {{"text": "Polish-Russian War", "type": "EVENT", "rank_score": 0.85}}
#             ],
#             "relations": [
#                 {{"subject": "John Smith", "predicate": "mother_of", "object": "Maria", "rank_score": 0.92}},
#                 {{"subject": "John Smith", "predicate": "directed_by", "object": "Polish-Russian War", "rank_score": 0.88}}
#             ]
#         }}
#         Query: {query}
#         Context: {text}

#         Expected output format (JSON):
#         """

#     else:
#         prompt = f"""
#         Below is a query. Extract key entities and relationships using step-by-step reasoning.
#         Rank the extracted entities based on their relevance to the query.

#         Query: {query}

#         Expected output format (JSON):
#         """

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
#     outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.9)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # 解析 JSON 输出
#     try:
#         structured_data = json.loads(response[len(prompt):])
#             # === 在这里加一层类型检查 ===
#         if not isinstance(structured_data, dict):
#             # LLM 可能输出的是list或别的类型，此时我们转成空dict，以免后续报错
#             structured_data = {"entities": [], "relations": []}

#         return structured_datareturn structured_data
#     except:
#         return {"entities": [], "relations": []}


if __name__ == "__main__":
    # 示例输入
    queries = [ "Who is the mother of the director of the film Polish-Russian War?",
        "Which country Audofleda's husband is from?",
        "Who is older, Dyson Parody or Gene Watson?"]
    query = "Who is the mother of the director of the film Polish-Russian War?"
    context = "The director of Polish-Russian War is John Smith, whose mother is Maria."
    model_name = "/data_vault/pittnail/yuj49/rag_reason/llama3.1_8B"  # 替换为实际 Llama 模型路径

    # 加载 Llama 模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # # 使用 Llama 提取关系（包含 context）
    # result_with_context = extract_relations_with_llama(model, tokenizer, query, text=context, include_context=True)
    # print("Extracted Entities and Relations with Context:")
    # print(result_with_context)

    # 使用 Llama 提取关系（仅包含 query）
    for query in queries:
        result_without_context = extract_relations_with_llama(model, tokenizer, query, include_context=False)
        print("*"*8)
        print("Extracted Entities and Relations without Context:")
        print(result_without_context)
        print("*"*8)
