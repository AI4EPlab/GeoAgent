import os, glob, json
import time
import re
import ast
from collections import defaultdict
import jsonlines



class FunctionCallCollector(ast.NodeVisitor):
    def __init__(self, global_function_set):
        self.calls = set()  # 用集合来存储函数调用，天然忽略顺序
        self.global_function_set = global_function_set

    def visit_Call(self, node):
        # 获取函数的名称
        function_name = None

        # 如果是 obj.method() 形式
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr  # 只保留方法名称
            function_name = f".{method_name}"
        # 如果是普通函数调用 func()
        elif isinstance(node.func, ast.Name):
            function_name = node.func.id

        # 过滤掉 unknown 和 print 函数调用
        if function_name and function_name != "print" and function_name != "unknown":
            self.calls.add(function_name)
            self.global_function_set.add(function_name)

        self.generic_visit(node)

def get_function_calls(code, global_function_set):
    """从代码中提取函数调用组合"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # 忽略语法错误的代码片段

    collector = FunctionCallCollector(global_function_set)
    collector.visit(tree)
    return frozenset(collector.calls)  # 返回不可变集合，方便用于字典键

class MethodCallFilter(ast.NodeVisitor):
    def __init__(self):
        self.contains_method_call = False

    def visit_Attribute(self, node):
        # 检查是否是方法调用
        if isinstance(node.ctx, ast.Load):
            # 找到对象.方法的形式
            self.contains_method_call = True
        # 继续遍历子节点
        self.generic_visit(node)


def filter_code_blocks_without_method_calls(code):
    # 将代码解析为 AST
    tree = ast.parse(code)
    filtered_code = []

    # 遍历 AST 中的每个节点
    for node in tree.body:

        # 创建过滤器，检查代码块是否包含方法调用
        visitor = MethodCallFilter()
        visitor.visit(node)

        # 如果代码块中有方法调用，则保留它
        if visitor.contains_method_call:
            filtered_code.append(node)

    #if filtered_code:
    #    print(ast.unparse(filtered_code))
    #    time.sleep(1)

    return filtered_code


if __name__ == '__main__':
    max_per_combination = 100
    """统计并保留每种函数组合最多 10 个片段"""
    combinations = defaultdict(list)
    global_function_set = set()

    with jsonlines.open('./ds1000-exec.jsonl', 'r') as f:
      for line in f:
        code = line['code']
        #if 'import ' in code:
        #  continue
        #if 'ee.batch.' in code:
        #    continue
        calls = get_function_calls(code, global_function_set)
        if calls:  # 如果成功解析出调用组合
            print(calls)
            combinations[calls].append(line)

    # 过滤每种组合，最多保留 10 个片段
    result = []
    for calls, snippets in combinations.items():
        result.extend(snippets[:max_per_combination])  # 每种组合只取前10个
    #print(result)
    print(global_function_set)
    print('we found ' + str(len(result)) + ' functions combinations')
    print('we called ' + str(len(global_function_set)) + ' functions')



