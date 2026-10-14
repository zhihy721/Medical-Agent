# 问诊工程评测政策 v1（待用户确认）

状态：`approved`。项目用户于 2026-09-25 确认本政策的工程口径；基线 Git `5acc4fe`。本政策用于评价 Medical-Agent 对**虚构中文问诊文本**的抽取、风险分层和下一步动作是否符合当前项目规则；不代表医生标注或临床有效性。政策确认不等于逐条场景已审查；只有逐条场景核对后，才允许把相应样本置为 `reviewed` 并冻结测试集。

## 1. 输入与来源

- 主任务输入：当前用户文字轮次。可另给显式的既有状态摘要，来源必须标明 `dialogue`、`manual_profile` 或 `profile_prefill`；已有状态不计作当前轮抽取正确。
- ASR 输出可作为文字输入演示，但 ASR 错误另列；TTS 不参与此政策的抽取或规划评分。
- 地理坐标和医院查询返回只影响医院列表展示，A 的风险/动作判断不依赖实时地理数据。配对测试使用固定 fixture。

## 2. 抽取标签

- `mention = (field, normalized_value, polarity, evidence, turn_id, start, end)`。`evidence` 必须是指定用户轮次的连续原文，Python 字符索引满足 `user[start:end] == evidence`。
- `polarity` 取 `present`、`negated`、`uncertain`。未提及不生成 mention；“没有胸痛”是 `negated` 的胸痛，不是空值，也不算现有胸痛。跨轮改口要保留两轮原始 mention，累计状态另记 superseded/correction。
- `field` 在 v1 采用 `chief_complaint`、`symptoms`、`accompanying_symptoms`、`duration`、`severity`、`location`、`city`、`age`、`sex`、`past_history`、`allergy_history`、`medication_history`、`red_flags` 和当前 `TCM_FIELD_PATTERNS` 的字段。`description` 是原文备份，不作为 mention 指标。
- 名称统一按 `tools/symptom_tool.py` 与 `knowledge/data/term_normalization.json` 的项目词表；不在冻结后为提高模型得分增加别名。允许人工标注文本中明显但规则未覆盖的表达，作为新模型相对基线的真实覆盖差异；字段与规范化值必须在冻结前核对。
- 在 v1 主指标中按 `(field, normalized_value, polarity)` 的每轮集合比较；证据位置单独计，不把已有画像字段重复计为 TP。当前规则/Mock 平面结果没有极性与证据时，报告为 `legacy_flat_proxy`，不得与以后专用结构化模型的证据支持率直接等同。

## 3. 风险政策

风险只按项目当前执行顺序解释：

1. `red_flags` 非空直接 HIGH，`matched_rules=["red_flag_detected"]`。
2. 否则按 `knowledge/data/risk_rules.json` 的列表顺序取**第一个**完全满足的规则。HIGH：胸痛+呼吸困难、胸痛+冠心病/高血压、重度、发热+咳嗽+呼吸困难、腹痛+呕吐/便血；MEDIUM：中/重度头痛+呕吐、发热+咳嗽、腹痛+胃病史、胸痛或呼吸困难、高龄（≥65）且有症状。实际先后顺序以该 JSON 为准，前面的 HIGH 可覆盖后面的 MEDIUM。
3. 无匹配规则但有症状为 LOW；无有效症状为 UNKNOWN。`LOW` 仅表示项目规则未命中。
4. 词表以外的疑似危险描述、含糊否认或医学处置争议标 `disputed`/`policy_gap`，不进入风险主分数；不得让教师/Judge 自动定义医学 gold。

## 4. 动作政策

当前七动作：`risk_escalation`、`clarify_conflict`、`ask_followup_single`、`ask_followup_bundle`、`summarize_progress`、`request_pulse_input`、`final_advice`。

- HIGH：只允许 `risk_escalation`，高于所有追问或脉诊。
- 非 HIGH、有未达澄清上限的冲突：`clarify_conflict`。同一冲突字段达到现有 2 次追问上限后，按当前规则进入后续流程，但记录 `unresolved_conflict` 以便 B 阶段安全门专项评估。
- 无冲突而有可问的缺失槽位：通常允许对应的单问/打包追问；达到现有进展总结条件时允许 `summarize_progress`。具体可接受动作由人工逐例核对的 `allowed_actions` 给出，不要求复刻规则 Planner 的唯一措辞。
- 无可问缺失槽位时：满足现有脉诊门槛且尚未询问/拒绝/已有脉象，可 `request_pulse_input`；否则 `final_advice`。信息极少的非 HIGH 场景不允许过早 `final_advice`。
- 工具出错、模型输出非法、网络超时时必须保留安全回退轨迹；未来 B 阶段 Gate 可能收紧，但 A 的 v1 gold 不在看到结果后悄悄改写。

## 5. 争议与版本

`draft`/`disputed` 不计正式冻结集主分数。标注人可以在 `notes` 说明争议，并决定只保留字段抽取 gold、留空风险 gold，或整例排除。确认后记录 `reviewed_by`、`reviewed_at` 和依据条款。数据/政策文件 SHA-256 与 Git commit 写入 manifest。冻结后若需改政策或标签，创建新版本并重跑所有对照，保留旧报告。
