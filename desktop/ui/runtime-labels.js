'use strict';
globalThis.medicalRuntimeLabel = mode => ({
  rule: '规则演示 · Mock抽取',
  sft: '4B旧抽取SFT · Qwen规划与检查',
  'g6-rule': 'G6事实抽取 · 规则规划与检查',
  'g6-plan': 'G6事实抽取 · Qwen规划 · 规则检查',
  'g6-review': 'G6事实抽取 · Qwen规划与检查',
  'full888-rule': '4B Full888抽取 · 规则规划与检查',
  'full888-plan': '4B Full888抽取 · Qwen规划 · 规则检查',
  'full888-review': '4B Full888抽取 · Qwen规划与检查',
  'full888-safe-rule': '4B Full888安全修复 · 规则规划与检查',
  'full888-safe-plan': '4B Full888安全修复 · Qwen规划 · 规则检查',
  'full888-safe-review': '4B Full888安全修复 · Qwen规划与检查',
  'qwen-prototype': 'Qwen联合意图 · 规划与执行前审查 · 三次输出原型',
  'qwen-grounded': 'Qwen规划与执行前评审 · 请求依据校验',
}[mode] || '未知运行配置');
