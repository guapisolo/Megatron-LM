#!/usr/bin/env python3
"""演示 MTP 的循环滚动策略"""

import torch

def demonstrate_roll_strategy():
    print("="*70)
    print("MTP 循环滚动策略演示")
    print("="*70)
    
    # 假设有一个句子 "The cat sat on mat"
    tokens = ["The", "cat", "sat", "on", "mat"]
    token_ids = torch.tensor([1, 2, 3, 4, 5])
    
    print("\n原始序列:")
    print(f"Tokens: {tokens}")
    print(f"IDs:    {token_ids.tolist()}")
    
    print("\n" + "="*70)
    print("标准 GPT 预测（不用 MTP）:")
    print("-"*70)
    print("Position 0: 输入 'The'  -> 预测 'cat'")
    print("Position 1: 输入 'cat'  -> 预测 'sat'")
    print("Position 2: 输入 'sat'  -> 预测 'on'")
    print("Position 3: 输入 'on'   -> 预测 'mat'")
    print("Position 4: 输入 'mat'  -> 预测 [下一个token]")
    
    print("\n" + "="*70)
    print("MTP with k=2 (预测未来2个token):")
    print("-"*70)
    
    # MTP Layer 0 (标准预测)
    print("\n【MTP Layer 0】标准预测:")
    print("Position 0: 输入 'The'  -> 预测 'cat'")
    print("Position 1: 输入 'cat'  -> 预测 'sat'")
    print("Position 2: 输入 'sat'  -> 预测 'on'")
    print("Position 3: 输入 'on'   -> 预测 'mat'")
    print("Position 4: 输入 'mat'  -> 预测 [next]")
    
    # MTP Layer 1 (roll -1)
    print("\n【MTP Layer 1】预测未来第2个token (roll -1):")
    rolled_1 = torch.roll(token_ids, shifts=-1)
    print(f"滚动后的输入: {rolled_1.tolist()} (原: {token_ids.tolist()})")
    print("Position 0: 输入 'cat'  -> 预测 'sat'  (原本在位置1)")
    print("Position 1: 输入 'sat'  -> 预测 'on'   (原本在位置2)")
    print("Position 2: 输入 'on'   -> 预测 'mat'  (原本在位置3)")
    print("Position 3: 输入 'mat'  -> 预测 [next] (原本在位置4)")
    print("Position 4: 输入 'The'  -> 预测 ???    (❌ 循环回来了!)")
    
    # MTP Layer 2 (roll -2 累计)
    print("\n【MTP Layer 2】预测未来第3个token (再roll -1):")
    rolled_2 = torch.roll(rolled_1, shifts=-1)
    print(f"滚动后的输入: {rolled_2.tolist()} (原: {token_ids.tolist()})")
    print("Position 0: 输入 'sat'  -> 预测 'on'   (原本在位置2)")
    print("Position 1: 输入 'on'   -> 预测 'mat'  (原本在位置3)")
    print("Position 2: 输入 'mat'  -> 预测 [next] (原本在位置4)")
    print("Position 3: 输入 'The'  -> 预测 ???    (❌ 循环回来了!)")
    print("Position 4: 输入 'cat'  -> 预测 ???    (❌ 循环回来了!)")
    
    print("\n" + "="*70)
    print("问题分析:")
    print("-"*70)
    print("1. torch.roll 是循环移位，末尾的元素会移到开头")
    print("2. 对于 k=2 的 MTP，最后 3 个位置会有问题：")
    print("   - Position 2,3,4 在某些 MTP layer 中会看到循环回来的token")
    print("   - 这些位置无法正确预测未来 k 个 tokens")
    print("\n3. 所以需要 mask 掉最后 k+1=3 个位置的 loss!")
    
    print("\n" + "="*70)
    print("Packed Sequences 的额外问题:")
    print("-"*70)
    print("假设打包了两个文档: [Doc1: 'The cat sat'] + [Doc2: 'I love dogs']")
    print("\nDoc1 的最后 3 个 tokens 在滚动后会错误地看到 Doc2 的内容：")
    print("- 'cat' 可能会用 'I' 来预测")
    print("- 'sat' 可能会用 'love' 来预测")
    print("\n这就是为什么 packed sequences 需要特殊的 mask 处理！")

if __name__ == "__main__":
    demonstrate_roll_strategy()