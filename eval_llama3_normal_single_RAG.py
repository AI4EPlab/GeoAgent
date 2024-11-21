import nest_asyncio
nest_asyncio.apply()
import logging
import sys
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SummaryIndex
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import QueryFusionRetriever

#llm
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
'''
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="microsoft/phi-2",
    model_name="microsoft/phi-2",
    device_map="cuda",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.bfloat16}
)
'''


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# loads BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

service_context = ServiceContext.from_defaults(
    chunk_size=2048,
    llm=None, #llm,
    embed_model=embed_model
)
set_global_service_context(service_context)

# this is very omportant and avoid the error in graph index
from llama_index.core import Settings
Settings.llm = None #llm
Settings.chunk_size = 2048
# maximum input size to the LLM
Settings.context_window = 4096
# number of tokens reserved for text generation.
Settings.num_output = 256


# infer using llama3
from unsloth import FastLanguageModel
import torch
import ast
import json
import jsonlines
import os
import re
import glob
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.bfloat16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
#model_name_path = 'gee-llama3-8B'
from multi_process import monitor_process, make_serializable, is_serializable
MAX_EXECUTION_TIME = 10  # 秒
MAX_MEMORY_USAGE = 90  # 百分比
from humaneval import stats_execute, get_prompts_with_ids, STOP_SEQUENCES
from human_eval.data import write_jsonl
from unsloth import FastLanguageModel
model_name_path = '/workspace/unsloth-llama-3.1-8b-Instruct'
model_name = '/workspace/unsloth-llama-3.1-8b-Instruct'
access_token = "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"
# Load the model and prepare it to be fine-tuned with QLoRA.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    cache_dir=model_name_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    rope_scaling='null',
)

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
# alpaca_prompt = You MUST copy from above!
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


def extract_code_split(text, pre_code):
    if '```' in text:
        new_text = []
        add_flag = False
        for text_i in text.splitlines(keepends=True):
            if add_flag and '```' not in text_i:
                new_text.append(text_i)
            if '```' in text_i:
                add_flag = not add_flag
        text = ''.join(new_text)
        # get ast code block
        code_block = ''
        # split code with '.' and '\n'
        multi_text = text.splitlines()
        multi_text = [text for text in multi_text if not text.startswith('#')]
        if multi_text:
            for i in range(len(multi_text)):
                try:
                    code_block = '\n'.join(multi_text)
                    ast.parse(code_block)
                    break
                except:
                    multi_text.pop()
                    code_block = '\n'.join(multi_text)
                    continue
    else:
        # get ast code block
        code_block = []
        one_block = []
        start_id = 0
        # split code with '.' and '\n'
        multi_text = text.splitlines()
        if '' in multi_text:
            multi_text.remove('')
        multi_text.append('end')

        if multi_text:
            for i, text_i in enumerate(multi_text):
                start_id += 1
                if start_id > 1:
                    if text_i.startswith(' '):
                        one_block.append(text_i)
                    else:
                        try:
                            one_block_ = '\n'.join(one_block) + '\n' + text_i
                            code_block.append('\n'.join(one_block))
                            one_block = []
                            ast.parse('\n'.join(code_block))
                            pop_one_block = False
                        except:
                            code_block.pop()
                            try:
                                code_block.append(one_block_)
                                ast.parse('\n'.join(code_block))
                                pop_one_block = True
                            except:
                                code_block.pop()
                                pop_one_block = False
                                pass
                        # test current line
                        try:
                            code_block.append(text_i)
                            one_block = []
                            ast.parse('\n'.join(code_block))
                            start_id = 0
                        except:
                            code_block.pop()
                            if not pop_one_block:
                                one_block.append(text_i)
                            continue
                else:
                    one_block.append(text_i)

        if 'end' in code_block:
            code_block.remove('end')
        code_block = '\n'.join(code_block)

    #print('----------------------------------------------------------------')
    #print(code_block)
    #print('----------------------------------------------------------------')
    #if pre_code not in code_block:
    #    code_block = pre_code + '\n' + code_block

    if code_block:
        new_context = ''
        unique_lines = []
        tree = ast.parse(code_block)
        line_to_nodes = sort_nodes_by_line(tree)
        for linen_key, line_values in line_to_nodes.items():
            line = max([ast.unparse(line_value) for line_value in line_values], key=len)
            if not is_string_contained(line, new_context):
                new_context = new_context + line
                unique_lines.append(line + '\n')

        selected_code_block = ''.join(unique_lines)

        error = None
    else:
        selected_code_block = ''
        error = 'Write right python code please!'
    return selected_code_block, error

def extract_code_again(text):
    if '```' in text:
        new_text = []
        add_flag = False
        for text_i in text.splitlines(keepends=True):
            if add_flag and '```' not in text_i:
                new_text.append(text_i)
            if '```' in text_i:
                add_flag = not add_flag
        text = ''.join(new_text)
        # get ast code block
        code_block = ''
        # split code with '.' and '\n'
        multi_text = text.splitlines()
        multi_text = [text for text in multi_text if not text.startswith('#')]
        if multi_text:
            for i in range(len(multi_text)):
                try:
                    code_block = '\n'.join(multi_text)
                    ast.parse(code_block)
                    break
                except:
                    multi_text.pop()
                    code_block = '\n'.join(multi_text)
                    continue
    else:
        # get ast code block
        code_block = []
        one_block = []
        start_id = 0
        # split code with '.' and '\n'
        multi_text = text.splitlines()
        if '' in multi_text:
            multi_text.remove('')
        multi_text.append('end')

        if multi_text:
            for i, text_i in enumerate(multi_text):
                start_id += 1
                if start_id > 1:
                    if text_i.startswith(' '):
                        one_block.append(text_i)
                    else:
                        try:
                            one_block_ = '\n'.join(one_block) + '\n' + text_i
                            code_block.append('\n'.join(one_block))
                            one_block = []
                            ast.parse('\n'.join(code_block))
                            pop_one_block = False
                        except:
                            code_block.pop()
                            try:
                                code_block.append(one_block_)
                                ast.parse('\n'.join(code_block))
                                pop_one_block = True
                            except:
                                code_block.pop()
                                pop_one_block = False
                                pass
                        # test current line
                        try:
                            code_block.append(text_i)
                            one_block = []
                            ast.parse('\n'.join(code_block))
                            start_id = 0
                        except:
                            code_block.pop()
                            if not pop_one_block:
                                one_block.append(text_i)
                            continue
                else:
                    one_block.append(text_i)

        if 'end' in code_block:
            code_block.remove('end')
        code_block = '\n'.join(code_block)

    if code_block:
        new_context = ''
        unique_lines = []
        tree = ast.parse(code_block)
        line_to_nodes = sort_nodes_by_line(tree)
        for linen_key, line_values in line_to_nodes.items():
            line = max([ast.unparse(line_value) for line_value in line_values], key=len)
            if not is_string_contained(line, new_context):
                new_context = new_context + line
                unique_lines.append(line + '\n')

        selected_code_block = ''.join(unique_lines)
    else:
        selected_code_block = ''
    return selected_code_block

def sort_nodes_by_line(node, line_to_nodes=None):
    """
    按照代码行的顺序对AST节点进行排列
    """
    if line_to_nodes is None:
        line_to_nodes = {}

    if hasattr(node, 'lineno'):
        line = node.lineno
        if line not in line_to_nodes:
            line_to_nodes[line] = []
        line_to_nodes[line].append(node)

    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    sort_nodes_by_line(item, line_to_nodes)
        elif isinstance(value, ast.AST):
            sort_nodes_by_line(value, line_to_nodes)

    return line_to_nodes


def is_string_contained(substring, string):
    # 删除两个字符串中的空格、换行符和单双引号
    cleaned_substring = re.sub(r'[\s\'"]', '', substring.strip()).replace('\\', '')
    cleaned_string = re.sub(r'[\s\'"]', '', string).replace('\\', '')
    return cleaned_substring in cleaned_string

def clean_code(code):
    message_prompt = [
        {"role": "system",
         "content": f"You goal is to clean the current and reorder thegiven Python code making it more logical and removing the repeated part."},
        {"role": "user",
         "content": f"Here is the current Python code: \n{code}" + f"\nYour answer:"}, ]
    input_prompt_ids = tokenizer.apply_chat_template(message_prompt, add_generation_prompt=True, return_tensors="pt").to(model.device)  # llama3.1
    prompt_outputs = model.generate(input_prompt_ids, max_new_tokens=512, eos_token_id=terminators, temperature=0.9)
    improved_code = tokenizer.decode(prompt_outputs[0][input_prompt_ids.shape[-1]:], skip_special_tokens=True)
    #print('----------------------fixed code--------------------------------')
    #print(improved_code)

    return improved_code

def update_prompt(prompt, library, rag_context):
    if library:
        message_prompt = [
            {"role": "system",
            "content": f"You goal is to choose right function based on {library} according to candidate functions but ingore the irrelavant ones."},
            {"role": "user",
            "content": f"here is the candidate functions: {rag_context}"
                    + f"You need to choose useful functions from candidates in prompt:{prompt}"
                    + "\n give function name or nothing related: {" + f"'function name': ..." + "}"}, ]
    else:
        message_prompt = [
            {"role": "system",
             "content": f"You goal is to choose right function according to candidate functions but ingore the irrelavant ones."},
            {"role": "user",
             "content": f"here is the candidate functions: {rag_context}"
                        + f"You need to add useful functions from candidates in prompt:{prompt}"
                        + "\n give function name or nothing related: {" + f"'function name': ..." + "}"}, ]
    input_prompt_ids = tokenizer.apply_chat_template(message_prompt, add_generation_prompt=True, return_tensors="pt").to(model.device)  # llama3.1
    prompt_outputs = model.generate(input_prompt_ids, max_new_tokens=512, eos_token_id=terminators, temperature=0.9)
    improved_prompt = tokenizer.decode(prompt_outputs[0][input_prompt_ids.shape[-1]:], skip_special_tokens=True)
    print('----------------------fixed code--------------------------------')
    print(improved_prompt)

    return improved_prompt

def model_generate_long_context(prompt, pre_tasks, rag_context, library, max_tokens, temperature, exec_path):
    work_path = f"import os\nos.chdir('{exec_path}')\nprint('the current work path is:', os.getcwd())\n"
    pre_code = pre_tasks.strip()
    prompt = prompt
    #print('pre_code', pre_code)
    if rag_context:
        if library:
            messages = [
                {"role": "system",
                 "content": f"I want you to become my Expert Python programmer. Your goal is to help me write python code for the given task using python library{library}"},
                {"role": "user",
                 "content": f"here is the possible functions: {rag_context}"
                            + f"You need to write code according to previous code {pre_code} and the detailed prompt:{prompt}"
                            + f"\n Give the your code:", }, ]
        else:
            messages = [
                {"role": "system",
                 "content": f"I want you to become my Expert Python programmer. Your goal is to help me write python code for the given task"},
                {"role": "user",
                 "content": f"You need to write code according to previous code {pre_code} and the detailed prompt:{prompt}"
                            + f"\n Give the your code:", }, ]
    else:
        if rag_context:
            messages = [
                {"role": "system",
                 "content": f"I want you to become my Expert Python programmer. Your goal is to help me write python code for the given task using python library {library}"},
                {"role": "user",
                 "content": f"here is the possible functions: {rag_context}"
                            + f"You need to write code according to the detailed prompt:{prompt}"
                            + "\n Give the your code:", }, ]
        else:
            messages = [
                {"role": "system",
                 "content": "I want you to become my Expert Python programmer. Your goal is to help me write python code for the given task"},
                {"role": "user",
                 "content": f"You need to write code according to the detailed prompt:{prompt}"
                            + "\n Give the your code:", }, ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=max_tokens, eos_token_id=terminators, temperature=temperature)
    code_i_initial = tokenizer.decode(outputs[0][input_ids.shape[-1]:])
    print('-----------------------initial code------------------------')
    print(code_i_initial)

    code_i, error = extract_code_split(code_i_initial, '')
    print('------------------------extracted code-----------------------')
    #code_i = clean_code(code_i)
    #code_i = extract_code_again(code_i)
    print(code_i)
    print('------------------------end code--------------------------')

    if code_i:
        cal_reward = 1
        #return_back = monitor_process(work_path + code_i, MAX_EXECUTION_TIME, MAX_MEMORY_USAGE)
        #if return_back:
        #    print('-------------------some error in long context--------------------')
        #    print(return_back)
    else:
        code_i = code_i_initial
        cal_reward = 0

    return code_i, 0, cal_reward


def beam_search(prompt, pre_tasks,  rag_context, library, exec_path):
    print('-----------------------------beam search------------------------')
    temperature = 0.9
    max_tokens = 1024
    tokens, probs, cal_reward = model_generate_long_context(prompt, pre_tasks,  rag_context, library, max_tokens=max_tokens, temperature=temperature, exec_path=exec_path)
    #print(tokens)
    return tokens, cal_reward


def calculate_reward(precode, completion, exec_path):
    work_path = f"import os\nos.chdir('{exec_path}')\nprint('the current work path is:', os.getcwd())\n"

    new_context = ''
    unique_lines = []
    # unique line
    tree = ast.parse(completion)
    line_to_nodes = sort_nodes_by_line(tree)
    for linen_key, line_values in line_to_nodes.items():
        line = max([ast.unparse(line_value) for line_value in line_values], key=len)
        if not is_string_contained(line, new_context):
            new_context = new_context + line
            unique_lines.append(line + '\n')

    code_len = len(unique_lines)
    test_code = ''
    if code_len == 0:
        return 0
    else:
        success_len = 0
        for code_i in unique_lines:
            test_code = test_code + '\n' + code_i
            return_back = monitor_process(work_path + test_code, MAX_EXECUTION_TIME, MAX_MEMORY_USAGE)
            if return_back is None:
                success_len += 1
            elif ('ERROR' not in return_back) and ('Error' not in return_back) and ('error' not in return_back):
                success_len += 1
            else:
                break
        return success_len / code_len

if __name__ == "__main__":
    base_dir = '/workspace/RAG_bench/GeoCode/'
    # get data input
    task_path = '/workspace/comparison_bench/GeoCode_GEE_selected.jsonl'
    # this is for cibench !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    exec_path_root = '/workspace/comparison_bench/GeoSpatial2task'
    output_doc = '/workspace/comparison_bench/GeoCode_GEE_selected_single.jsonl'
    task_name = os.path.basename(task_path).split('.')[0]

    tasks = []
    ori_task_id = 0
    with jsonlines.open(task_path, mode='r') as eval_reader:
        for sample in eval_reader:
            if 'task' in sample:
                task = sample['task']
            elif 'prompt' in sample:
                task = sample['prompt']
            else:
                task = None
            if 'context' in sample:
                pre_task = sample['context']
            elif 'pre_code' in sample:
                pre_task = sample['pre_code']
            else:
                pre_task = None

            if 'exec_path' in sample:
                exec_path_ = sample['exec_path']
            else:
                exec_path_ = None

            if 'functions' in sample:
                functions = sample['functions']
            else:
                functions = None

            if 'exlib' in sample:
                library = sample['exlib']
            elif 'libraries' in sample:
                library = sample['libraries']
            else:
                library = None
            new_library = []
            if library is not None:
                for lib in library:
                    if 'import' not in lib:
                        new_library.append(lib)
                    elif 'from' in lib:
                        lib = lib.split('import')[0].split('from')[1].strip()
                        if '.' in lib:
                            lib = lib.split('.')[0].strip()
                        new_library.append(lib)
                    elif 'import' in lib:
                        lib = lib.split('import')[1].strip()
                        if 'as' in lib:
                            lib = lib.split('as')[0].strip()
                        if '.' in lib:
                            lib = lib.split('.')[0].strip()
                        new_library.append(lib)
                new_library = list(set(new_library))

            if pre_task is not None:
                tasks.append({'task': task, 'code': sample['code'], 'library': new_library, 'pre_task': pre_task,
                              'ori_task_id': ori_task_id, 'exec_path': exec_path_, 'functions': functions})
            else:
                _pre_task = extract_code_again(sample['prompt'])
                tasks.append({'task': task, 'code': sample['code'], 'library': new_library, 'pre_task': _pre_task,
                              'ori_task_id': ori_task_id, 'exec_path': exec_path_, 'functions': functions})
            ori_task_id += 1

    # cut tasks according to the output
    len_tasks = len(tasks)
    if os.path.exists(output_doc):
        with open(output_doc, 'r', encoding='utf-8') as file:
            # Read all lines
            lines = file.readlines()
            if lines:
                # Get the last line
                last_line = lines[-1]
                # Parse the JSON object
                try:
                    last_json = json.loads(last_line)
                    current_line = last_json['ori_task_id']
                except:
                    current_line = len_tasks
                    time.sleep(100)
    else:
        current_line = len_tasks

    tasks = tasks[0:current_line]

    num_iter = 1
    # processing and add item dynamically
    task_id = current_line

    while tasks:
        # print tasks
        print('----------------------start task---------------------------------')
        current_task = tasks.pop()  # select the last one
        if 'ori_task_id' in current_task:
            ori_task_id = current_task['ori_task_id']

        if 'pre_task' in current_task:
            pre_tasks = current_task['pre_task']

        if 'functions' in current_task:
            functions = current_task['functions']
            functions = [function.split('.')[-1] for function in functions]
        else:
            functions = None

        # save for next step replacing failed task in multi-step tasks
        if 'code' in current_task:
            code = current_task['code']
        else:
            code = None

        if 'library' in current_task:
            library = current_task['library']
            print('here is the library!', library)
        else:
            library = None

        if current_task['exec_path'] is not None:
            exec_path = os.path.join(exec_path_root, current_task['exec_path'])
        else:
            exec_path = exec_path_root

        if not pre_tasks.endswith('\n'): pre_tasks = pre_tasks + '\n'
        prompt = current_task['task'].replace('prompt', '').replace('{', '').replace('}', '').replace(':', '').replace("''", '')

        task_id = task_id + 1
        print(f"---- STARTING Coding FOR {str(task_id)} ({num_iter}/{len(tasks)}) ----")
        print(prompt)
        # --------------------------------------------RAG----------------------------
        if library is not None:
            rag_dir_list = [base_dir + lib for lib in library]
            rag_dir_list_exist = all(os.path.exists(rag_dir) for rag_dir in rag_dir_list)
        else:
            rag_dir_list_exist = False

        rag_context = []
        if rag_dir_list_exist:
            for lib in library:
                rag_dir = base_dir + lib
                filelist = glob.glob(os.path.join(rag_dir, '*'))
                file_nm = [os.path.basename(nm) for nm in filelist]
                library_list = [nm.split('.')[0] for nm in file_nm]
                library_metadatas = {}
                for library in library_list:
                    library_metadatas[library] = {'library': library}  # it may can change to description of the library
                # load all documents
                docs_dict = []
                for doc_nm in file_nm:
                    library = doc_nm.split('.')[0]
                    doc = SimpleDirectoryReader(
                        input_files=[f"{rag_dir}/{doc_nm}"]
                    ).load_data()[0]
                    doc.metadata.update(library_metadatas[library])
                    docs_dict.append(doc)
                # build vector
                # simple vector store
                vector_store = SimpleVectorStore()
                vector_storage_context = StorageContext.from_defaults(vector_store=vector_store)
                indexes = []
                for doc in docs_dict:
                    vector_index = VectorStoreIndex.from_documents(
                        [doc], service_context=service_context, storage_context=vector_storage_context
                    )
                    indexes.append(vector_index)
                # build retriver
                indexes_retriever = [idx.as_retriever() for idx in indexes]
                if len(indexes_retriever) < 1:
                    continue
                try:
                    retriever = QueryFusionRetriever(
                        indexes_retriever,
                        similarity_top_k=3,
                        num_queries=1,  # set this to 1 to disable query generation
                        use_async=True,
                        verbose=True,
                    )
                except:
                    continue
                # run recursive retriever
                rag_outputs = retriever.retrieve(prompt)
                for rag_output in rag_outputs:
                    rag_output_text = rag_output.node.get_content()
                    if len(rag_output_text) > 4000:
                        rag_output_text = rag_output_text[0:4000]
                    rag_context.append(rag_output_text)
                print('------------------------------------rag start-------------------------------------------------')
                print('rag_context', rag_context)
                print('------------------------------------rag end-------------------------------------------------')
                del docs_dict, vector_store, vector_storage_context, indexes, vector_index, indexes_retriever, retriever

        print('final reward evaluation!!!')
        if rag_context:
            for func_ in functions:
                prompt = prompt.replace(func_, '')
            #rag_context = update_prompt(prompt, library, rag_context)
        completion, cal_reward = beam_search(prompt, '', rag_context, library, exec_path)
        if cal_reward:
            # reward = calculate_reward(pre_tasks, completion, exec_path)
            reward = -999
        else:
            reward = -1

        item = dict(
            task_id=task_id,
            ori_task_id=ori_task_id,
            reward=reward,
            gen_code=completion,
            prompt=prompt,
            code=code,
            pre_tasks=pre_tasks,
            stats=dict(
            ),
        )

        write_jsonl(f"{task_name}_single.jsonl", [item], append=True)
        print(f"---- COMPLETED MCTS FOR {str(task_id)} ({num_iter}/{len(tasks)}) ----")

