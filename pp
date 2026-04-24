python compare_gemma4.py --baseline_dir ./baseline_dump --test_dir ./npu_dump

# 放宽容差（比如NPU精度稍低）
python compare_gemma4.py --baseline_dir ./baseline_dump --test_dir ./npu_dump --rtol 1e-03 --atol 1e-05
