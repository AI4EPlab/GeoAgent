import json
import jsonlines
import os
import ast
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
import numpy as np

# Parse the AST tree to extract function names and arguments
def ast_func_name(node):
    # stop call
    if isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Call):
            function_name = ast_func_name(node.func.value)
        else:
            dump = ast.dump(node)
            function_name = dump.split(", args=[]")[0].split("func=")[1]
            if function_name.startswith("Name(id='"):
                function_name = function_name.split("Name(id='")[1].split("', ctx=Load()")[0]
            elif "value=Name(id='" in function_name:
                function_name = function_name.split("value=Name(id='")[1].split("ctx=Load()")
                function_rest = function_name[1:]
                function_id = function_name[0].split("'")[0]
                function_attr = []
                for attr in function_rest:
                    if "attr='" in attr:
                        function_attr.append(attr.split("attr='")[1].split("'")[0])
                function_name = ".".join([function_id]+function_attr)
    else:
        # Figure out function name
        dump = ast.dump(node)
        function_name = dump.split(", args=[]")[0].split("func=")[1]
        if function_name.startswith("Name(id='"):
            function_name = function_name.split("Name(id='")[1].split("', ctx=Load()")[0]
        elif "value=Name(id='" in function_name:
            function_name = function_name.split("value=Name(id='")[1].split("ctx=Load()")
            function_rest = function_name[1:]
            function_id = function_name[0].split("'")[0]
            function_attr = []
            for attr in function_rest:
                if "attr='" in attr:
                    function_attr.append(attr.split("attr='")[1].split("'")[0])
            function_name = ".".join([function_id]+function_attr)
    return function_name

def split_calls_assign(target, call_node):
    nodes = []
    call_chain = []

    # Traverse the chain of calls
    while isinstance(call_node, ast.Call):
        call_chain.append(call_node)
        if isinstance(call_node.func, ast.Attribute):
            call_node = call_node.func.value
        else:
            break

    # Handle the initial call
    initial_call = call_chain.pop()
    nodes.append(ast.Assign(
        targets=[target],
        value=initial_call
    ))

    # Handle the subsequent calls
    for call in reversed(call_chain):
        # print(call)
        # print(ast.dump(call))
        new_value = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=target.id, ctx=ast.Load()),
                attr=call.func.attr,
                ctx=ast.Load()
            ),
            args=call.args,
            keywords=call.keywords
        )
        nodes.append(ast.Assign(
            targets=[target],
            value=new_value
        ))

    return nodes


def fetch_all_func(content):
    func_calls = []
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            nodes = split_calls_assign(ast.Name(id='temp'), node)
            #print([ast.dump(node) for node in nodes])
            for node_i in nodes:
                if isinstance(node_i.value, ast.Call):
                    try:
                        func_ = ast_func_name(node_i.value)
                        if not 'Attribute' in func_:
                            func_calls.append(func_)
                    except Exception as e:
                        print('-----------------wrong calll-----------------------')
                        print(ast.dump(node_i.value))
                        print(e)
    return func_calls


 # 自定义准确率计算方法
def example_based_accuracy(y_true_bin, y_pred_bin):
    correct = 0
    total = len(y_true_bin)

    for true, pred in zip(y_true_bin, y_pred_bin):
        intersection = sum(t & p for t, p in zip(true, pred))
        union = sum(t | p for t, p in zip(true, pred))
        if union == 0:
            accuracy = 1  # 当没有任何标签时，认为准确率为1
        else:
            accuracy = intersection / union
        correct += accuracy

    return correct / total

def Hamming_Loss(y_true, y_pred):
    temp = 0
    stat = 0
    for i in range(len(y_true)):
        temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
        stat += len(y_true[i])
    return temp / stat


def Recall(y_true, y_pred):
    temp = 0
    for i in range(len(y_true)):
        if sum(y_true[i]) == 0:
            continue
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    return temp / len(y_true)


def Precision(y_true, y_pred):
    temp = 0
    for i in range(len(y_true)):
        if sum(y_pred[i]) == 0:
            continue
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return temp / len(y_true)


def F1Measure(y_true, y_pred):
    temp = 0
    for i in range(len(y_true)):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp += (2 * sum(np.logical_and(y_true[i], y_pred[i]))) / (sum(y_true[i]) + sum(y_pred[i]))
    return temp /len(y_true)


if __name__ == '__main__':
    y_true = []
    y_pred = []
    y_true_bin = []
    y_pred_bin = []

    #/home/yusin/docker_folder/comparison_bench/GeoCode_Other_selected_mcts.jsonl
    #/home/yusin/docker_folder/comparison_bench/single_call/llama3.1/GeoCode_Other_selected_single.jsonl
    with (jsonlines.open('/home/yusin/docker_folder/comparison_bench/single_call/gemma/cibench-exec_single.jsonl', mode='r') as eval_reader):
        samples = [sample for sample in eval_reader]
        idx_sample = 0
        for i in range(len(samples)):
            sample = samples[i]
            if sample['ori_task_id'] == 'insert':
                continue

            idx_sample += 1
            if idx_sample > 100:
                break

            global_function_set = set()

            label = sample['code']
            #if 'pre_tasks' in sample:
            #    label = sample['pre_tasks'] + '\n' + label
            # fetch functions from the text
            try:
                #label = get_function_calls(label, global_function_set)
                label = list(set(fetch_all_func(label)))
            except:
                label = []
            if label is None: label = []
            if 'print' in label: label.remove('print')
            #if 'Authenticate' in label: label.remove('Authenticate')
            #if 'Initialize' in label: label.remove('Initialize')
            label = list(set([label_i.split('.')[-1] if '.' in label_i else label_i for label_i in label]))
            if label:
                y_true.append(label)
            else:
                continue

            infer = sample['gen_code']
            #if 'pre_tasks' in sample:
            #    if sample['pre_tasks'] in infer:
            #        infer = infer.replace(sample['pre_tasks'], '')
            # fetch functions from the text
            try:
                #infer = get_function_calls(infer, global_function_set)
                infer = fetch_all_func(infer)
                #func_pre = fetch_all_func(sample['pre_tasks'])
                #print('infer:', infer)
                #print('pre:', func_pre)
                #for func_ in func_pre:
                #    infer.remove(func_)
                infer = list(set(infer))
                #print('infer again:', infer)
            except:
                infer = []
            if infer is None: infer = []
            if 'print' in infer: infer.remove('print')
            #if 'Authenticate' in infer: infer.remove('Authenticate')
            #if 'Initialize' in infer: infer.remove('Initialize')
            infer = list(set([infer_i.split('.')[-1] if '.' in infer_i else infer_i for infer_i in infer]))
            infer_nan = ['None' for _ in range(len(label))]
            idx = 0
            for lbl in infer:
                if lbl in label:
                    infer_nan[idx] = lbl
                    idx += 1

            #if idx == 0:
            #    continue
            y_pred.append(infer_nan)

            print('y_true:', label)
            print('y_pred:', infer_nan)


            # 构建所有可能标签的全集 sample based
            all_labels = sorted(set(label_ for label_ in label + infer_nan))
            # 将预测结果和真实标签转换为多标签二进制矩阵
            y_true_bin.append([1 if label_ in sorted(label) else 0 for label_ in all_labels])
            y_pred_bin.append([1 if label_ in sorted(infer_nan) else 0 for label_ in all_labels])


        #print('y_true:', y_true)
        #print('y_pred:', y_pred)


    #print(y_true_bin)
    #print(y_pred_bin)
    # 计算评价指标
    accuracy = example_based_accuracy(y_true_bin, y_pred_bin)
    precision = Precision(y_true_bin, y_pred_bin)
    recall = Recall(y_true_bin, y_pred_bin)
    f1 = F1Measure(y_true_bin, y_pred_bin)
    hamming = Hamming_Loss(y_true_bin, y_pred_bin)

    # 打印结果
    print(f'Accuracy (example-based): {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Hamming Loss: {hamming}')
