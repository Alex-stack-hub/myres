import os
import argparse
import torch
from collections import defaultdict
from typing import Dict, Tuple, Optional

def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                    rtol: float = 1e-05, atol: float = 1e-08) -> Dict:
    """对比两个Tensor，返回差异指标"""
    result = {
        "allclose": False,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "shape_match": True,
        "dtype_match": True
    }
    
    # 检查形状和类型
    if tensor1.shape != tensor2.shape:
        result["shape_match"] = False
        result["shape1"] = tuple(tensor1.shape)
        result["shape2"] = tuple(tensor2.shape)
        return result
    
    if tensor1.dtype != tensor2.dtype:
        result["dtype_match"] = False
        result["dtype1"] = str(tensor1.dtype)
        result["dtype2"] = str(tensor2.dtype)
        # 继续对比数值，类型不同也能看差异
    
    # 计算误差
    try:
        abs_error = torch.abs(tensor1 - tensor2)
        result["max_abs_error"] = abs_error.max().item()
        result["mean_abs_error"] = abs_error.mean().item()
        result["allclose"] = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    except Exception as e:
        result["error"] = str(e)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Gemma4 Dump文件精度对比工具")
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
    
    # 检查目录是否存在
    if not os.path.exists(args.baseline_dir):
        print(f"❌ 基线目录不存在: {args.baseline_dir}")
        return
    if not os.path.exists(args.test_dir):
        print(f"❌ 测试目录不存在: {args.test_dir}")
        return
    
    # 获取两个目录下的所有pt文件
    baseline_files = set([f for f in os.listdir(args.baseline_dir) if f.endswith(".pt")])
    test_files = set([f for f in os.listdir(args.test_dir) if f.endswith(".pt")])
    
    common_files = baseline_files & test_files
    only_baseline = baseline_files - test_files
    only_test = test_files - baseline_files
    
    # 开始对比
    print("="*80)
    print("📊 Gemma4 精度对比工具")
    print(f"基线目录: {args.baseline_dir}")
    print(f"测试目录: {args.test_dir}")
    print(f"容差设置: rtol={args.rtol}, atol={args.atol}")
    print("="*80)
    
    results = []
    layer_results = defaultdict(dict)
    
    for filename in common_files:
        baseline_path = os.path.join(args.baseline_dir, filename)
        test_path = os.path.join(args.test_dir, filename)
        
        try:
            tensor_baseline = torch.load(baseline_path, map_location="cpu")
            tensor_test = torch.load(test_path, map_location="cpu")
        except Exception as e:
            print(f"⚠️ 加载文件失败 {filename}: {e}")
            continue
        
        # 对比
        comp_result = compare_tensors(tensor_baseline, tensor_test, args.rtol, args.atol)
        comp_result["filename"] = filename
        results.append(comp_result)
        
        # 按层分类（如果是层文件）
        if filename.startswith("layer"):
            layer_idx = filename.split("_")[0].replace("layer", "")
            if layer_idx not in layer_results:
                layer_results[layer_idx] = {"files": [], "max_error": 0.0}
            layer_results[layer_idx]["files"].append(comp_result)
            if comp_result["max_abs_error"] > layer_results[layer_idx]["max_error"]:
                layer_results[layer_idx]["max_error"] = comp_result["max_abs_error"]
    
    # 生成报告
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("📊 Gemma4 精度差异报告")
    report_lines.append("="*80)
    report_lines.append(f"基线目录: {args.baseline_dir}")
    report_lines.append(f"测试目录: {args.test_dir}")
    report_lines.append(f"容差设置: rtol={args.rtol}, atol={args.atol}")
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
        report_lines.append(f"  ⚠️ 仅基线存在的文件: {len(only_baseline)} (如: {list(only_baseline)[:3]})")
    if only_test:
        report_lines.append(f"  ⚠️ 仅测试存在的文件: {len(only_test)} (如: {list(only_test)[:3]})")
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
                line += f" | ⚠️ 形状不匹配: {r['shape1']} vs {r['shape2']}"
            else:
                line += f" | MaxAbsErr: {r['max_abs_error']:.6e} | MeanAbsErr: {r['mean_abs_error']:.6e}"
            report_lines.append(line)
    else:
        report_lines.append("未找到Layer 0的文件")
    report_lines.append("")
    
    # 按差异大小排序的所有文件（Top 20 差异最大）
    report_lines.append("="*80)
    report_lines.append("🔥 差异最大的文件 (Top 20)")
    report_lines.append("="*80)
    sorted_results = sorted(results, key=lambda x: x.get("max_abs_error", 0), reverse=True)
    for i, r in enumerate(sorted_results[:20]):
        status = "✅" if r["allclose"] else "❌"
        line = f"{i+1:2d}. {status} {r['filename']}"
        if not r["shape_match"]:
            line += f" | ⚠️ 形状不匹配"
        else:
            line += f" | MaxAbsErr: {r['max_abs_error']:.6e} | MeanAbsErr: {r['mean_abs_error']:.6e}"
        report_lines.append(line)
    report_lines.append("")
    
    # 按层汇总的最大误差
    report_lines.append("="*80)
    report_lines.append("📚 按层汇总的最大误差")
    report_lines.append("="*80)
    sorted_layers = sorted(layer_results.items(), key=lambda x: float(x[0]) if x[0].replace("-", "").isdigit() else float('inf'))
    for layer_idx, data in sorted_layers:
        max_err = data["max_error"]
        # 简单标记：误差>1e-3标红，>1e-5标黄
        if max_err > 1e-3:
            marker = "🔴"
        elif max_err > 1e-5:
            marker = "🟡"
        else:
            marker = "🟢"
        report_lines.append(f"{marker} Layer {layer_idx}: MaxAbsErr = {max_err:.6e}")
    
    # 保存报告
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    # 打印到终端
    print("\n".join(report_lines))
    print("\n" + "="*80)
    print(f"✅ 对比完成！详细报告已保存至: {args.output}")
    print("="*80)

if __name__ == "__main__":
    main()
