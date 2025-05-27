# Load model from local directory
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# 本地模型路径
local_model_path = "/home/chegan/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B"

# 全局加载模型和分词器，以便函数调用时无需重复加载
print("正在加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)
print("模型和分词器已加载！")

def generate_streamed_response(prompt_text: str, enable_thinking: bool = False):
    """
    使用加载好的模型和分词器，对输入的文本进行流式生成回答。

    Args:
        prompt_text (str): 用户输入的问题或提示。
        enable_thinking (bool): 是否启用思考模式，默认为False。
    """
    print(f"\n我的问题 (思考模式: {enable_thinking}): {prompt_text}")
    print("模型的回答 (流式): ")

    # 1. 初始化 TextStreamer
    #    - tokenizer: 用于解码生成的token
    #    - skip_prompt: True 表示在流式输出时跳过原始的prompt文本
    #    - skip_special_tokens: True 跳过特殊token，如 <|endoftext|>
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 2. 将问题文本编码为模型输入
    inputs = tokenizer(prompt_text, return_tensors="pt")

    # 3. 根据模式设置不同的生成参数
    if enable_thinking:
        generation_kwargs = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            # "min_p": 0, # min_p 并非Hugging Face Transformers generate方法的标准参数，通常top_p和top_k已能很好控制采样
        }
        print("使用思考模式参数: temperature=0.6, top_p=0.95, top_k=20")
    else:
        generation_kwargs = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            # "min_p": 0,
        }
        print("使用非思考模式参数: temperature=0.7, top_p=0.8, top_k=20")

    # 通用生成参数
    common_generation_kwargs = {
        "streamer": streamer,
        "max_new_tokens": 500,
        "do_sample": True, # 确保不使用贪婪解码
        "repetition_penalty": 1.1 # 减少重复
        # 如果模型支持 presence_penalty 并且您想用它替代或补充 repetition_penalty，可以在这里添加
        # "presence_penalty": 0.0, # 示例值，0表示不惩罚
    }

    # 合并参数
    final_generation_kwargs = {**inputs, **common_generation_kwargs, **generation_kwargs}

    _ = model.generate(**final_generation_kwargs)
    
    # 流式输出结束后，可以加一个换行符，使后续输出更清晰
    print()


if __name__ == '__main__':
    generate_streamed_response("请给我写一个快速排序的Python代码。", enable_thinking=True)

