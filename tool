import os
import argparse
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, Tuple, Optional

def compute_cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> Optional[float]:
    """计算两个Tensor的余弦相似度（展平成一维后计算）"""
    if tensor1.shape != tensor2.shape:
        return None
    try:
        t1_flat = tensor1.flatten()
        t2_flat = tensor2.flatten()
        cos_sim = F.cosine_similarity(t1_flat.unsqueeze(0), t2_flat.unsqueeze(0), dim=1)
        return cos_sim.item()
    except Exception as e:
        return None

def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                    rtol: float = 1e-05, atol: float = 1e-08) -> Dict:
    """对比两个Tensor，返回差异指标（含余弦相似度）"""
    result = {
        "allclose": False,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "cosine_similarity": None,
        "shape_match": True,
        "dtype_match": True
    }
    
    if tensor1.shape != tensor2.shape:
        result["shape_match"] = False
        result["shape1"] = tuple(tensor1.shape)
        result["shape2"] = tuple(tensor2.shape)
        return result
    
    if tensor1.dtype != tensor2.dtype:
        result["dtype_match"] = False
        result["dtype1"] = str(tensor1.dtype)
        result["dtype2"] = str(tensor2.dtype)
    
    result["cosine_similarity"] = compute_cosine_similarity(tensor1, tensor2)
    
    try:
        abs_error = torch.abs(tensor1 - tensor2)
        result["max_abs_error"] = abs_error.max().item()
        result["mean_abs_error"] = abs_error.mean().item()
        result["allclose"] = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    except Exception as e:
        result["error"] = str(e)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Gemma4 Dump文件精度对比工具（重点标注输入层）")
    parser.add_argument("--baseline_dir", type=str, required=True, 
                        help="基线（GPU/正确版本）的Dump目录路径")
    parser.add_argument("--test_dir", type=str, required=True, 
                        help="测试版（NPU/待验证）的Dump目录路径")
    parser.add_argument("--rtol", type=float, default=1e-05, 
                        help="allclose的相对容差，默认1e-05")
    parser.add_argument("--atol", type=float, default=1e-08, 
                        help="allclose的绝对容差，默认1e-08")
    parser.add_argument("--output", type=str, default="gemma4_diff_report.txt",
                        help="差异报告保存路径，默认gemma4_diff_report.txt")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.baseline_dir):
        print(f"❌ 基线目录不存在: {args.baseline_dir}")
        return
    if not os.path.exists(args.test_dir):
        print(f"❌ 测试目录不存在: {args.test_dir}")
        return
    
    baseline_files = set([f for f in os.listdir(args.baseline_dir) if f.endswith(".pt")])
    test_files = set([f for f in os.listdir(args.test_dir) if f.endswith(".pt")])
    
    common_files = baseline_files & test_files
    only_baseline = baseline_files - test_files
    only_test = test_files - baseline_files
    
    print("="*80)
    print("📊 Gemma4 精度对比工具（重点标注输入层）")
    print(f"基线目录: {args.baseline_dir}")
    print(f"测试目录: {args.test_dir}")
    print(f"容差设置: rtol={args.rtol}, atol={args.atol}")
    print("="*80)
    
    results = []
    layer_results = defaultdict(dict)
    input_layer_results = {}  # 专门存储输入层结果
    
    for filename in common_files:
        baseline_path = os.path.join(args.baseline_dir, filename)
        test_path = os.path.join(args.test_dir, filename)
        
        try:
            tensor_baseline = torch.load(baseline_path, map_location="cpu")
            tensor_test = torch.load(test_path, map_location="cpu")
        except Exception as e:
            print(f"⚠️ 加载文件失败 {filename}: {e}")
            continue
        
        comp_result = compare_tensors(tensor_baseline, tensor_test, args.rtol, args.atol)
        comp_result["filename"] = filename
        results.append(comp_result)
        
        # 专门收集输入层结果
        if "global_model_input_ids" in filename or "global_model_embedding_output" in filename:
            input_layer_results[filename] = comp_result
        
        if filename.startswith("layer"):
            layer_idx = filename.split("_")[0].replace("layer", "")
            if layer_idx not in layer_results:
                layer_results[layer_idx] = {"files": [], "max_error": 0.0, "min_cos_sim": 1.0}
            layer_results[layer_idx]["files"].append(comp_result)
            if comp_result["max_abs_error"] > layer_results[layer_idx]["max_error"]:
                layer_results[layer_idx]["max_error"] = comp_result["max_abs_error"]
            if comp_result["cosine_similarity"] is not None and comp_result["cosine_similarity"] < layer_results[layer_idx]["min_cos_sim"]:
                layer_results[layer_idx]["min_cos_sim"] = comp_result["cosine_similarity"]
    
    # 生成报告
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("📊 Gemma4 精度差异报告（重点标注输入层）")
    report_lines.append("="*80)
    report_lines.append(f"基线目录: {args.baseline_dir}")
    report_lines.append(f"测试目录: {args.test_dir}")
    report_lines.append(f"容差设置: rtol={args.rtol}, atol={args.atol}")
    report_lines.append("")
    report_lines.append("📌 指标说明:")
    report_lines.append("  - MaxAbsErr: 最大绝对误差（越小越好）")
    report_lines.append("  - MeanAbsErr: 平均绝对误差（越小越好）")
    report_lines.append("  - CosineSim: 余弦相似度（越接近1越好，范围[-1,1]）")
    report_lines.append("")
    
    # ====================== 【新增：最高优先级输入层验证】 ======================
    report_lines.append("="*80)
    report_lines.append("🔴 🔴 🔴 最高优先级：输入层验证 🔴 🔴 🔴")
    report_lines.append("="*80)
    report_lines.append("⚠️  重要提示：输入层理论上应该完全一致，不应该有任何误差！")
    report_lines.append("")
    
    # 1. input_ids 验证
    input_ids_file = "global_model_input_ids.pt"
    if input_ids_file in input_layer_results:
        r = input_layer_results[input_ids_file]
        report_lines.append("1️⃣  原始Token ID (global_model_input_ids.pt)")
        report_lines.append("   理论要求：✅ 必须完全一致 (torch.equal == True)")
        
        if r["shape_match"]:
            # 专门检查是否完全相等
            try:
                t1 = torch.load(os.path.join(args.baseline_dir, input_ids_file), map_location="cpu")
                t2 = torch.load(os.path.join(args.test_dir, input_ids_file), map_location="cpu")
                is_exact_equal = torch.equal(t1, t2)
                if is_exact_equal:
                    report_lines.append("   ✅✅✅ Token ID 完全一致！输入没问题！")
                else:
                    report_lines.append("   ❌❌❌ Token ID 不一致！！！问题出在Tokenizer/输入处理！")
                    report_lines.append(f"   基线Shape: {tuple(t1.shape)} | 测试Shape: {tuple(t2.shape)}")
            except Exception as e:
                report_lines.append(f"   ⚠️  检查完全相等失败: {e}")
        else:
            report_lines.append(f"   ❌❌❌ 形状不匹配！基线: {r['shape1']} vs 测试: {r['shape2']}")
    else:
        report_lines.append("1️⃣  原始Token ID (global_model_input_ids.pt)")
        report_lines.append("   ⚠️  文件不存在，无法验证！")
    
    report_lines.append("")
    
    # 2. embedding_output 验证
    embed_file = "global_model_embedding_output.pt"
    if embed_file in input_layer_results:
        r = input_layer_results[embed_file]
        report_lines.append("2️⃣  词嵌入层输出 (global_model_embedding_output.pt)")
        report_lines.append("   理论要求：✅ 误差极小 (MaxAbsErr < 1e-6, CosineSim > 0.999999)")
        
        status = "✅" if r["allclose"] else "❌"
        if not r["shape_match"]:
            report_lines.append(f"   {status} 形状不匹配！基线: {r['shape1']} vs 测试: {r['shape2']}")
        else:
            cos_sim_str = f"{r['cosine_similarity']:.8f}" if r["cosine_similarity"] is not None else "N/A"
            
            # 针对Embedding的特殊判断
            if r["cosine_similarity"] is not None and r["cosine_similarity"] > 0.999999 and r["max_abs_error"] < 1e-6:
                embed_status = "✅✅✅ 词嵌入层一致性极好！"
            elif r["cosine_similarity"] is not None and r["cosine_similarity"] > 0.999:
                embed_status = "🟡 词嵌入层有轻微误差，可能是精度舍入问题"
            else:
                embed_status = "❌❌❌ 词嵌入层误差很大！问题出在Embedding层/权重加载！"
            
            report_lines.append(f"   {status} {embed_status}")
            report_lines.append(f"   MaxAbsErr: {r['max_abs_error']:.8e} | MeanAbsErr: {r['mean_abs_error']:.8e} | CosineSim: {cos_sim_str}")
    else:
        report_lines.append("2️⃣  词嵌入层输出 (global_model_embedding_output.pt)")
        report_lines.append("   ⚠️  文件不存在，无法验证！")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("")
    
    # 统计信息
    total_files = len(common_files)
    passed_files = sum(1 for r in results if r["allclose"])
    failed_files = total_files - passed_files
    
    report_lines.append(f"📈 统计汇总:")
    report_lines.append(f"  共同文件数: {total_files}")
    report_lines.append(f"  ✅ 通过allclose: {passed_files}")
    report_lines.append(f"  ❌ 未通过allclose: {failed_files}")
    if only_baseline:
        report_lines.append(f"  ⚠️  仅基线存在的文件: {len(only_baseline)} (如: {list(only_baseline)[:3]})")
    if only_test:
        report_lines.append(f"  ⚠️  仅测试存在的文件: {len(only_test)} (如: {list(only_test)[:3]})")
    report_lines.append("")
    
    # 重点关注：第一层
    report_lines.append("="*80)
    report_lines.append("🎯 重点关注：第一层 (Layer 0)")
    report_lines.append("="*80)
    layer0_files = [r for r in results if "layer0_" in r["filename"] or "LAYER0_FOCUS" in r["filename"]]
    if layer0_files:
        for r in sorted(layer0_files, key=lambda x: x["filename"]):
            status = "✅" if r["allclose"] else "❌"
            line = f"{status} {r['filename']}"
            if not r["shape_match"]:
                line += f" | ⚠️  形状不匹配: {r['shape1']} vs {r['shape2']}"
            else:
                cos_sim_str = f"{r['cosine_similarity']:.6f}" if r["cosine_similarity"] is not None else "N/A"
                line += f" | MaxAbsErr: {r['max_abs_error']:.6e} | MeanAbsErr: {r['mean_abs_error']:.6e} | CosineSim: {cos_sim_str}"
            report_lines.append(line)
    else:
        report_lines.append("未找到Layer 0的文件")
    report_lines.append("")
    
    # 按余弦相似度排序的文件（最不相似的Top 20）
    report_lines.append("="*80)
    report_lines.append("🔥 余弦相似度最低的文件 (Top 20)")
    report_lines.append("="*80)
    sorted_by_cos = sorted([r for r in results if r["cosine_similarity"] is not None], 
                          key=lambda x: x["cosine_similarity"])
    for i, r in enumerate(sorted_by_cos[:20]):
        status = "✅" if r["allclose"] else "❌"
        cos_sim_str = f"{r['cosine_similarity']:.6f}"
        line = f"{i+1:2d}. {status} {r['filename']} | CosineSim: {cos_sim_str}"
        if r["shape_match"]:
            line += f" | MaxAbsErr: {r['max_abs_error']:.6e}"
        report_lines.append(line)
    report_lines.append("")
    
    # 按差异大小排序的所有文件（Top 20 差异最大）
    report_lines.append("="*80)
    report_lines.append("🔥 最大绝对误差最大的文件 (Top 20)")
    report_lines.append("="*80)
    sorted_results = sorted(results, key=lambda x: x.get("max_abs_error", 0), reverse=True)
    for i, r in enumerate(sorted_results[:20]):
        status = "✅" if r["allclose"] else "❌"
        line = f"{i+1:2d}. {status} {r['filename']}"
        if not r["shape_match"]:
            line += f" | ⚠️  形状不匹配"
        else:
            cos_sim_str = f"{r['cosine_similarity']:.6f}" if r["cosine_similarity"] is not None else "N/A"
            line += f" | MaxAbsErr: {r['max_abs_error']:.6e} | MeanAbsErr: {r['mean_abs_error']:.6e} | CosineSim: {cos_sim_str}"
        report_lines.append(line)
    report_lines.append("")
    
    # 按层汇总的最大误差和最小余弦相似度
    report_lines.append("="*80)
    report_lines.append("📚 按层汇总（最大误差 & 最小余弦相似度）")
    report_lines.append("="*80)
    sorted_layers = sorted(layer_results.items(), key=lambda x: float(x[0]) if x[0].replace("-", "").isdigit() else float('inf'))
    for layer_idx, data in sorted_layers:
        max_err = data["max_error"]
        min_cos_sim = data["min_cos_sim"]
        
        err_marker = "🔴" if max_err > 1e-3 else ("🟡" if max_err > 1e-5 else "🟢")
        cos_marker = "🔴" if min_cos_sim < 0.999 else ("🟡" if min_cos_sim < 0.9999 else "🟢")
        
        cos_sim_str = f"{min_cos_sim:.6f}" if min_cos_sim != 1.0 else "N/A"
        report_lines.append(f"{err_marker}{cos_marker} Layer {layer_idx}: MaxAbsErr = {max_err:.6e} | MinCosineSim = {cos_sim_str}")
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print("\n".join(report_lines))
    print("\n" + "="*80)
    print(f"✅ 对比完成！详细报告已保存至: {args.output}")
    print("="*80)

if __name__ == "__main__":
    main()
