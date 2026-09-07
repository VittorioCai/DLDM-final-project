# 数据与评估必须通过的测试

在任何完整训练之前，把下列检查写成 `pytest` 自动测试，而不是只肉眼抽查。

1. `test_total_size_constant`：所有预算训练量均为 50,000。
2. `test_digit_balance`：每个数字均为 5,000 张。
3. `test_exact_counterexample_budget`：实际 conflict 数严格等于 N，且每类为 N/10。
4. `test_color_marginal_constant`：各预算全局十种颜色计数完全一致。
5. `test_nested_counterexamples`：相同 seed 下，小预算冲突索引是大预算子集。
6. `test_seed_reproducibility`：相同 seed 结果逐项相同；不同 seed 至少索引排列不同。
7. `test_counterfactual_shape_invariance`：换色前后灰度 mask、标签和样本 id 不变。
8. `test_all_hues_rendered_once`：每个测试形状恰好生成十种颜色版本。
9. `test_resnet_output_shape`：输入 `[B,3,28,28]`，输出 `[B,10]`。
10. `test_metric_known_examples`：用人工构造预测验证 flip rate、consistency、worst-group 的公式。

此外，训练前随机保存一张 10×10 图像网格：十个数字 × 十种颜色。人眼确认数字形状、标签、颜色映射和背景处理均正确。

