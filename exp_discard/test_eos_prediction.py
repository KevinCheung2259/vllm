#!/usr/bin/env python3
"""
测试EOS token概率预测实验的脚本

这个脚本用于测试修改后的vLLM调度器，验证是否能够正确记录每次迭代的EOS token概率。
使用chat模式进行测试。
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Any

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid
from transformers import AutoTokenizer


def setup_environment():
    """设置环境变量"""
    os.environ['VLLM_ENABLE_EOS_PROB_LOGGING'] = 'true'
    os.environ['VLLM_EOS_PROB_LOG_FILE'] = 'eos_probabilities.jsonl'
    print("环境变量设置完成:")
    print(f"  VLLM_ENABLE_EOS_PROB_LOGGING = {os.environ['VLLM_ENABLE_EOS_PROB_LOGGING']}")
    print(f"  VLLM_EOS_PROB_LOG_FILE = {os.environ['VLLM_EOS_PROB_LOG_FILE']}")


def clear_log_file():
    """清空之前的日志文件"""
    log_file = os.environ.get('VLLM_EOS_PROB_LOG_FILE', 'eos_probabilities.jsonl')
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"已清空之前的日志文件: {log_file}")


def create_test_prompts() -> List[List[Dict[str, str]]]:
    """创建测试对话提示"""
    prompts = [
        # 对话1: 写诗
        [
            {"role": "user", "content": "写一首关于春天的短诗。"}
        ],
        # 对话2: 解释概念
        [
            {"role": "user", "content": "解释什么是机器学习。"}
        ],
        # 对话3: 讲故事
        [
            {"role": "user", "content": "讲一个短故事。"}
        ],
        # 对话4: 数学问题
        [
            {"role": "user", "content": "1+1等于多少？"}
        ],
        # 对话5: 描述
        [
            {"role": "user", "content": "描述一下你最喜欢的季节。"}
        ],
        # 对话6: 多轮对话
        [
            {"role": "user", "content": "你好，请介绍一下自己。"},
            {"role": "assistant", "content": "你好！我是一个AI助手，很高兴为您服务。"},
            {"role": "user", "content": "你能帮我写代码吗？"}
        ]
    ]
    return prompts


async def run_inference_test():
    """运行推理测试"""
    print("\n开始推理测试...")
    
    # 设置HF_TOKEN环境变量
    os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN'
    
    # 初始化模型（使用较小的模型进行测试）
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    os.environ['VLLM_FORCED_EOS_ID'] = str(tokenizer.eos_token_id)
    
    try:
        # 使用AsyncLLMEngine进行chat模式推理
        engine_args = AsyncEngineArgs(
            model=model_name,
            max_model_len=512,
            dtype="float16",
            trust_remote_code=True,
            max_num_seqs=10,
            max_num_batched_tokens=2048
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print(f"成功加载模型: {model_name}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保模型可用，或修改model_name为其他可用模型")
        return

    # 创建采样参数，确保启用logprobs
    sampling_params = SamplingParams(
        max_tokens=200,
        temperature=0.8,
        logprobs=20,  # 启用logprobs，获取top-20 tokens的概率
        include_stop_str_in_output=False
    )

    # 获取测试对话
    conversations = create_test_prompts()
    
    print(f"\n准备进行{len(conversations)}个对话的推理...")
    print("采样参数:")
    print(f"  max_tokens: {sampling_params.max_tokens}")
    print(f"  temperature: {sampling_params.temperature}")
    print(f"  logprobs: {sampling_params.logprobs}")
    
    # 执行推理
    start_time = time.time()

    async def collect_one(idx: int, request_id: str, prompt: str):
        last_out = None
        async for out in engine.generate(prompt=prompt,
                                         sampling_params=sampling_params,
                                         request_id=request_id):
            last_out = out
        return last_out
    
    tasks = []
    request_ids = []
    for i, conversation in enumerate(conversations):
        request_id = f"req_{i}_{random_uuid()}"
        request_ids.append(request_id)
        # 转换对话为prompt字符串（确保返回字符串而不是token ids）
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        assert isinstance(prompt, str), f"apply_chat_template未返回字符串，得到: {type(prompt)}"
        tasks.append(asyncio.create_task(collect_one(i, request_id, prompt)))

    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    print(f"\n推理完成! 用时: {end_time - start_time:.2f}秒")
    
    # 显示生成结果
    print("\n=== 生成结果 ===")
    for i, output in enumerate(results):
        if output is None:
            print(f"\n对话 {i+1}: (无输出)")
            continue
        print(f"\n对话 {i+1}:")
        print(f"请求ID: {output.request_id}")
        print(f"生成文本: {output.outputs[0].text}")
        print(f"生成token数: {len(output.outputs[0].token_ids)}")
        print(f"完成原因: {output.outputs[0].finish_reason}")
        
        # 显示logprobs信息（如果有）
        if hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs:
            print(f"Logprobs数量: {len(output.outputs[0].logprobs)}")


def analyze_eos_probabilities():
    """分析EOS token概率日志"""
    log_file = os.environ.get('VLLM_EOS_PROB_LOG_FILE', 'eos_probabilities.jsonl')
    
    if not os.path.exists(log_file):
        print(f"\n未找到EOS概率日志文件: {log_file}")
        return
    
    print(f"\n=== 分析EOS概率日志: {log_file} ===")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            entries = [json.loads(line.strip()) for line in f if line.strip()]
        
        if not entries:
            print("日志文件为空")
            return
        
        print(f"总共记录了 {len(entries)} 条EOS概率数据")
        
        # 按request_id分组分析
        request_data = {}
        for entry in entries:
            req_id = entry['request_id']
            if req_id not in request_data:
                request_data[req_id] = []
            request_data[req_id].append(entry)
        
        print(f"涉及 {len(request_data)} 个请求")
        
        # 分析每个请求的EOS概率变化
        for req_id, data in request_data.items():
            data.sort(key=lambda x: x['step'])  # 按步骤排序
            
            print(f"\n请求 {req_id}:")
            print(f"  提示长度: {data[0]['prompt_length']}")
            print(f"  生成步骤数: {len(data)}")
            print(f"  最终状态: {data[-1]['finish_reason']}")
            
            # 显示EOS概率变化趋势
            eos_probs = [d.get('eos_prob') for d in data]
            valid_probs = [p for p in eos_probs if p is not None]
            
            if valid_probs:
                print(f"  EOS概率范围: {min(valid_probs):.4f} - {max(valid_probs):.4f}")
                print(f"  平均EOS概率: {sum(valid_probs)/len(valid_probs):.4f}")
                
                # # 显示前几步和最后几步的EOS概率
                # print("  前5步EOS概率:", end="")
                # for i, prob in enumerate(eos_probs[:5]):
                #     print(f" {prob:.4f}" if prob is not None else " None", end="")
                # print()
                
                # if len(eos_probs) > 5:
                #     print("  后5步EOS概率:", end="")
                #     for i, prob in enumerate(eos_probs[-5:]):
                #         print(f" {prob:.4f}" if prob is not None else " None", end="")
                #     print()
                # 显示每一步的EOS概率
                print("  每一步EOS概率:", end="")
                for i, prob in enumerate(eos_probs):
                    print(f" {prob:.4f}" if prob is not None else " None", end="")
                print()
            else:
                print("  未记录到有效的EOS概率数据")
        
        # 总体统计
        all_valid_probs = []
        final_step_probs = []
        
        for req_id, data in request_data.items():
            for entry in data:
                if entry.get('eos_prob') is not None:
                    all_valid_probs.append(entry['eos_prob'])
            
            # 最后一步的概率
            if data and data[-1].get('eos_prob') is not None:
                final_step_probs.append(data[-1]['eos_prob'])
        
        if all_valid_probs:
            print(f"\n=== 总体统计 ===")
            print(f"有效EOS概率记录数: {len(all_valid_probs)}")
            print(f"EOS概率全局范围: {min(all_valid_probs):.4f} - {max(all_valid_probs):.4f}")
            print(f"全局平均EOS概率: {sum(all_valid_probs)/len(all_valid_probs):.4f}")
            
            if final_step_probs:
                print(f"最终步骤平均EOS概率: {sum(final_step_probs)/len(final_step_probs):.4f}")
        
    except Exception as e:
        print(f"分析日志文件时出错: {e}")


async def main():
    """主函数"""
    print("=== vLLM EOS Token概率预测实验 (Chat模式) ===")
    print("这个脚本将测试修改后的vLLM调度器记录EOS token概率的功能")
    
    # 设置环境
    setup_environment()
    
    # 清空之前的日志
    clear_log_file()
    
    # 运行推理测试
    await run_inference_test()
    
    # 等待一下确保日志写入完成
    await asyncio.sleep(2)
    
    # 分析结果
    analyze_eos_probabilities()
    
    print("\n=== 实验完成 ===")
    print("您可以查看 eos_probabilities.jsonl 文件获取详细的EOS概率数据")
    print("这些数据可以用于进一步分析和建模，预测请求何时会结束")


if __name__ == "__main__":
    asyncio.run(main()) 