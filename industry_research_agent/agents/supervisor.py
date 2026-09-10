"""Deterministic supervisor for evidence-based research routing."""
from __future__ import annotations
import re

from langchain_core.messages import AIMessage, HumanMessage

from state import DIMENSIONS, ResearchState
from triage import detect_region, required_dimensions, triage_request
from routing import classify_request


DIMENSION_TO_ANALYST = {
    "市场": "market_analyst",
    "趋势": "market_analyst",
    "机会": "market_analyst",
    "竞争": "competition_analyst",
    "风险": "competition_analyst",
    "商业模式": "business_analyst",
}
ROUTE_PRIORITY = ["市场", "竞争", "商业模式", "趋势", "机会", "风险"]
MAX_AGENT_STEPS = 10

KNOWN_INDUSTRIES = {
    "AI创业": ("AI初创", "ai初创", "人工智能创业", "AI创业", "ai创业", "AI公司", "ai公司"),
    "pet_economy": ("宠物", "宠物烘焙", "宠物食品"),
    "coffee": ("咖啡", "咖啡店", "咖啡馆"),
    "second_hand_luxury": ("二手奢侈品", "中古", "奢侈品回收"),
    "catering": ("餐饮", "餐厅", "外卖", "小吃"),
    "education": ("教培", "培训", "教育"),
    "e_commerce": ("电商", "直播带货", "跨境电商"),
}


def latest_user_text(state: ResearchState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return ""


def identify_industry(text: str) -> str:
    # Preserve mixed English/Chinese long-tail names such as "AI教育" instead
    # of collapsing them into the broad education category.
    mixed = re.search(r"([A-Za-z][A-Za-z0-9-]*[\u4e00-\u9fff]{1,12})(?:行业|赛道|市场|生意|电商|店)", text)
    if mixed:
        return mixed.group(1)
    for industry, keywords in KNOWN_INDUSTRIES.items():
        if any(keyword in text for keyword in keywords):
            return industry
    # Long-tail fallback: preserve the user's own noun phrase instead of forcing a
    # closed taxonomy. Very vague requests remain unknown and trigger a question.
    vague = {"做", "做生意", "创业", "投资", "行业", "项目", "不知道", "随便"}
    candidates = re.findall(r"([\u4e00-\u9fffA-Za-z0-9]{2,18})(?:行业|赛道|市场|生意|电商|店)", text)
    for candidate in candidates:
        candidate = re.sub(r"^(?:(?:我想|想做|准备做|了解|调研))+", "", candidate)
        candidate = re.sub(r"(行业|赛道|市场|生意|电商|店)$", "", candidate)
        if len(candidate) >= 2 and candidate not in vague:
            return candidate
    topic = re.search(r"(?:20\d{2}年?)?([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9-]{1,15})(?:有哪些|有什么|目前|最新)(?:政策|趋势|机会|风险|公司|设备)", text)
    if topic:
        return topic.group(1)
    return "unknown"


def extract_profile(state: ResearchState) -> tuple[str, str, str]:
    text = latest_user_text(state)
    industry = state.get("industry", "unknown")
    if industry == "unknown":
        industry = identify_industry(text)
    # Answers to the initial scope questionnaire are stored in known_facts.
    # Use a concrete free-text answer as the object on the next turn, but do
    # not mistake broad category buttons for a real industry.
    if industry == "unknown":
        facts = state.get("known_facts", {}) or {}
        broad_scope = {"开店/本地生意", "互联网或AI产品", "制造业/工厂", "研究某家公司"}
        for key in ("industry_detail", "company_name", "industry", "request_scope"):
            candidate = str(facts.get(key, "")).strip()
            if candidate and candidate not in broad_scope and candidate not in {"暂不确定", "还没确定"}:
                industry = candidate
                break

    region = state.get("region", "全国") or "全国"
    detected_region = detect_region(text)
    if detected_region:
        region = detected_region

    budget = state.get("budget", "未知")
    budget_range = re.search(r"(\d+(?:\.\d+)?)\s*万\s*[～~至到-]\s*(\d+(?:\.\d+)?)\s*万", text)
    budget_match = re.search(r"(?:预算|投入|投资)?\s*(\d+(?:\.\d+)?)\s*万", text)
    if budget_range:
        budget = f"{budget_range.group(1)}万～{budget_range.group(2)}万"
    elif budget_match:
        budget = f"{budget_match.group(1)}万"
    return industry, region, budget


def identify_intent(text: str) -> str:
    if any(word in text for word in ("实际回本", "实际ROI", "我的店", "日均", "已经开", "运营数据")):
        return "roi_calculation"
    if any(word in text for word in ("诊断", "亏损", "优化", "复购", "经营")):
        return "business_diagnosis"
    if any(word in text for word in ("对比", "哪个好", "比较")):
        return "industry_comparison"
    if any(word in text for word in ("市场规模", "行业趋势", "市场怎么样")):
        return "market_overview"
    return "entry_feasibility"


def identify_business_stage(text: str) -> str:
    if any(word in text for word in ("已经开", "正在运营", "我的店", "日均", "上个月", "实际")):
        return "operating"
    if any(word in text for word in ("筹备", "计划", "选址", "店型", "加盟")):
        return "planning"
    if any(word in text for word in (
        "想开", "想做", "值不值得", "还没开始", "没有开始", "准备开",
        "初创公司", "创业公司", "成立公司", "创办公司", "注册公司", "开始创业",
    )):
        return "pre_launch"
    return "unknown"


def build_research_plan(intent: str, stage: str) -> list[dict]:
    if intent == "market_overview":
        return [{"dimensions": ["市场", "趋势", "机会"], "mode": "evidence"}]
    if intent == "roi_calculation" and stage == "operating":
        return [{"dimensions": ["商业模式"], "mode": "personalized_roi"}]
    if stage in ("pre_launch", "planning"):
        return [
            {"dimensions": ["市场", "趋势", "机会"], "mode": "evidence"},
            {"dimensions": ["竞争", "风险"], "mode": "evidence"},
            {"dimensions": ["商业模式"], "mode": "scenario"},
        ]
    return [{"dimensions": ["市场", "竞争", "商业模式"], "mode": "evidence"}]


def build_interview_question(industry: str, region: str, budget: str, intent: str) -> str:
    """Ask one high-value context question before spending search/LLM budget."""
    if intent == "industry_comparison":
        return "你准备比较哪些具体行业或方案？请列出 2–3 个，以及最看重的指标（收益、风险、投入或增长）。"
    if region in ("全国", "未知", "") and intent in ("entry_feasibility", "market_overview"):
        return f"你想把“{industry}”落在哪个城市或区域？如果还没确定，我可以先按全国数据分析。"
    if budget in ("未知", "") and intent == "entry_feasibility":
        return f"你对“{industry}”的预算大概是多少？如果还没确定，我可以按小预算/中等预算/高预算三档估算。"
    if intent == "business_diagnosis":
        return "你目前最想解决哪个经营问题？例如获客、复购、成本、选址、产品结构或竞争压力。"
    if industry in ("pet_economy", "coffee", "catering"):
        return "你更倾向哪种经营形态？例如社区门店、商场店、工作室、外卖/线上，或暂时还没决定。"
    return f"你做这次“{industry}”调研最想支持哪个决定？例如是否进入、怎么进入、选址、产品方向或预算配置。"


def _question(topic: str, title: str, question: str, why: str, options: list[str]) -> dict:
    return {"id": topic, "topic": topic, "title": title, "question": question, "why": why,
            "options": options, "allow_custom": True, "skip_label": "暂不确定",
            "start_label": "结束访谈，开始调研"}


def _interview_consent_question(target_questions: int) -> dict:
    return {
        "id": "interview_consent",
        "topic": "interview_consent",
        "title": "访谈模式",
        "question": "是否先进入访谈模式，再开始联网调研？",
        "why": f"预计询问 {max(3, target_questions)} 个会影响结论的问题；回答越具体，建议越贴合你的实际情况。",
        "options": ["进入访谈（推荐）", "直接开始调研"],
        "allow_custom": False,
        "skip_label": "",
        "start_label": "",
    }


def _scenario_question_bank(scenario: str, industry: str, region: str, budget: str) -> list[dict]:
    common_goal = _question(
        "decision_goal", "本次目标", "这次你最希望解决哪个决定？",
        "目标不同，后续搜索范围和建议标准会完全不同。",
        ["判断是否值得做", "确定怎么开始", "比较几个方案", "识别主要风险"],
    )
    digital_idea = any(term in industry.lower() for term in (
        "ai", "人工智能", "软件", "互联网", "saas", "智能体", "agent",
    ))
    idea_budget = _question(
        "budget", "投入预算", "第一阶段可以投入多少预算？", "预算决定验证范围、团队规模和可承受的试错周期。",
        (["20万元以内", "20万～100万元", "100万～500万元", "500万元以上"] if digital_idea
         else ["10万元以内", "10万～30万元", "30万～100万元", "100万元以上"]),
    )
    idea_form = (_question(
        "business_form", "产品方向", "你准备做哪类 AI 产品或服务？", "产品方向决定客户、团队能力和城市资源需求。",
        ["企业软件/SaaS", "面向个人的应用", "AI+垂直行业解决方案", "模型/开发者工具"],
    ) if digital_idea else _question(
        "business_form", "经营形式", "你更倾向哪种落地形式？", "经营形式决定成本结构和竞争对手。",
        ["社区门店", "商场/核心商圈", "工作室/小型场地", "线上或轻资产模式"],
    ))
    idea_resources = (_question(
        "resources", "核心资源", "你目前已经具备哪些创业资源？", "已有技术、行业和客户资源会改变城市选择与启动路径。",
        ["技术/算法能力", "行业经验和场景", "潜在客户/渠道", "尚在组队"],
    ) if digital_idea else _question(
        "resources", "已有资源", "你目前具备哪些资源？", "已有资源会显著降低进入成本和试错风险。",
        ["相关行业经验", "客户或私域渠道", "供应商/合作伙伴", "目前都没有"],
    ))
    idea_customer = _question(
        "target_customer", "目标客户", "你最想服务哪类客户？", "目标客户决定产品、定价和获客渠道。",
        (["大型企业", "中小企业", "个人用户", "尚未确定"] if digital_idea
         else ["大众刚需用户", "中高端用户", "企业客户", "暂未确定"]),
    )
    idea_involvement = (_question(
        "involvement", "团队阶段", "目前团队处于什么阶段？", "团队结构会影响研发速度、融资需求和适合落地的城市。",
        ["已有联合创始人", "已有核心团队", "正在组队", "个人探索"],
    ) if digital_idea else _question(
        "involvement", "投入方式", "你准备亲自经营到什么程度？", "是否亲自经营会改变人工成本和执行风险。",
        ["全职亲自经营", "兼职参与", "招聘团队经营", "暂未确定"],
    ))
    idea_success = _question(
        "success", "成功标准", "什么结果会让你认为值得继续投入？", "需要明确后续继续或停止的判断标准。",
        (["获得首批付费客户", "验证稳定使用需求", "完成融资", "形成可复制获客"] if digital_idea
         else ["先验证真实需求", "半年内达到盈亏平衡", "形成稳定客户群", "能够复制扩张"]),
    )
    banks = {
        "idea": [
            common_goal,
            *([] if region not in ("全国", "未知", "") else [_question(
                "region", "目标地区", "你准备在哪个城市或区域尝试？", "地区会改变需求、租金和竞争基准。",
                ["一线城市", "新一线/二线城市", "三四线城市", "暂未确定"],
            )]),
            *([] if budget not in ("未知", "") else [idea_budget]),
            idea_form,
            idea_resources,
            idea_customer,
            idea_involvement,
            idea_success,
        ],
        "planning": [
            common_goal,
            _question("customer", "目标客户", "你已经确定的目标客户是谁？", "客户画像决定选址、产品和价格。",
                      ["大众消费者", "中高端消费者", "企业客户", "还没确定"]),
            _question("options", "候选方案", "你目前在比较哪些地点、产品或方案？", "明确候选项后才能使用统一口径比较。",
                      ["已有明确候选项", "只有一个初步方案", "仍在收集方案"]),
            _question("budget_allocation", "预算约束", "预算中最不能超支的是哪一部分？", "硬约束会改变落地路径。",
                      ["场地和装修", "设备和产品", "人员成本", "流动资金"]),
            _question("timeline", "决策时间", "你最晚什么时候需要做决定或启动？", "时间会影响数据时效和方案复杂度。",
                      ["1个月内", "1～3个月", "3～6个月", "没有明确期限"]),
            _question("channel", "渠道能力", "你已有客户渠道或供应链吗？", "渠道成熟度决定获客和交付风险。",
                      ["两者都有", "只有客户渠道", "只有供应链", "目前都没有"]),
            _question("success", "验收标准", "启动后最重要的成功指标是什么？", "建议必须围绕可衡量结果。",
                      ["销售收入", "客户数量", "毛利或回本", "市场验证"]),
        ],
        "operating_diagnosis": [
            _question("problem", "核心问题", "目前最需要解决的问题是什么？", "先聚焦一个主要问题才能建立诊断假设。",
                      ["销量/订单下降", "成本上升", "利润下降", "客户流失或复购低"]),
            _question("timing", "发生时间", "问题从什么时候开始明显出现？", "时间点有助于寻找外部变化和内部事件。",
                      ["最近1个月", "1～3个月", "3～12个月", "长期存在"]),
            _question("metric", "指标变化", "哪个关键指标变化最明显？", "指标决定需要验证的原因。",
                      ["客流/线索", "转化率", "客单价", "毛利率/成本"]),
            _question("actions", "已尝试措施", "你已经尝试过哪些改进？", "避免重复无效建议。",
                      ["降价促销", "增加投放", "调整产品", "还没系统尝试"]),
            _question("constraints", "现实约束", "目前最难改变的约束是什么？", "方案必须在真实约束内可执行。",
                      ["预算", "人员", "场地/产能", "供应链"]),
            _question("success", "改善目标", "你希望多长时间内看到什么改善？", "目标决定行动优先级和验证周期。",
                      ["一个月内止跌", "三个月改善利润", "半年恢复增长", "先找出根因"]),
        ],
        "existing_product_extension": [
            _question("current_capability", "现有能力", "现有设备、产线或软件能力处于什么水平？", "决定局部改造还是重新建设。",
                      ["设备较新且有余量", "设备可用但产能紧张", "设备老旧", "暂时无法判断"]),
            _question("bottleneck", "当前瓶颈", "现在最主要的瓶颈是什么？", "瓶颈决定投资应优先落在哪个环节。",
                      ["产能不足", "质量不稳定", "人工效率低", "无法承接新产品"]),
            _question("target", "新增目标", "新增能力主要服务什么需求？", "需求类型会改变技术路线和设备规格。",
                      ["已有确定订单", "新产品开发", "降低成本", "提升自动化"]),
            _question("production", "生产特征", "订单批量和产品规格稳定吗？", "柔性生产和大批量生产需要不同方案。",
                      ["大批量且稳定", "小批量多品种", "季节性波动", "还不确定"]),
            _question("budget", "改造预算", "本次扩展的预算级别是？", "预算决定改造深度和供应商范围。",
                      ["小规模试点", "单工位/局部改造", "整线升级", "预算待评估"]),
            _question("downtime", "停产约束", "可以接受多长时间停产改造？", "停产窗口是实施路线的硬约束。",
                      ["不能停产", "1周以内", "1个月以内", "可以分阶段实施"]),
            _question("team", "人员能力", "现有团队能否维护新增设备或系统？", "维护能力影响供应商和自动化程度。",
                      ["已有专业团队", "可以培训", "依赖供应商", "暂未评估"]),
            _question("success", "验收指标", "改造成功最重要的指标是什么？", "用于比较方案并设定验收门槛。",
                      ["产能提升", "单位成本下降", "质量稳定", "交付周期缩短"]),
        ],
        "company_research": [
            _question("scope", "研究范围", "希望聚焦哪条业务线、产品或技术？", "范围越明确，证据和结论越可靠。",
                      ["公司整体", "具体业务线", "产品/设备清单", "某项技术"]),
            _question("focus", "关注重点", "你最关心哪个方面？", "决定报告重点和来源优先级。",
                      ["技术能力", "供应链", "竞争格局", "商业机会"]),
            _question("period", "时间范围", "研究哪个时间范围？", "公司产品和竞争状态变化很快。",
                      ["当前最新", "近一年", "近三年变化", "指定历史阶段"]),
            _question("comparison", "比较对象", "需要与哪些竞争对手比较？", "对比能让结论更有决策意义。",
                      ["不需要比较", "国内竞争者", "国际竞争者", "我会自行填写"]),
        ],
        "option_comparison": [
            _question("options", "候选方案", "你要比较哪些具体选项？", "没有明确候选项无法建立统一比较。",
                      ["两个行业", "两个城市", "两个供应商", "两种技术路线"]),
            _question("priority", "决策权重", "你最看重哪两个指标？", "权重决定推荐结果。",
                      ["收益与增长", "成本与回本", "风险与稳定", "时间与落地难度"]),
            _question("constraint", "硬性约束", "哪项限制绝对不能突破？", "先排除不可执行方案。",
                      ["预算", "时间", "人员能力", "合规风险"]),
            _question("horizon", "决策周期", "你更关注短期还是长期结果？", "时间周期会改变优劣判断。",
                      ["半年以内", "1～2年", "3年以上", "短期长期都要"]),
        ],
    }
    return banks.get(scenario, [common_goal])


def build_scenario_question(state: ResearchState, scenario: str, industry: str, region: str, budget: str, intent: str) -> dict:
    """Return the next unanswered questionnaire item for the active scenario."""
    covered = set(state.get("covered_topics", []))
    for candidate in _scenario_question_bank(scenario, industry, region, budget):
        if candidate["topic"] not in covered:
            return candidate
    return {}


def supervisor_node(state: ResearchState) -> dict:
    industry, region, budget = extract_profile(state)
    latest = latest_user_text(state)
    prior_stage = state.get("business_stage", "unknown")
    inferred_stage = identify_business_stage(latest)
    stage = inferred_stage if inferred_stage != "unknown" else prior_stage
    all_user_text = " ".join(str(message.content) for message in state.get("messages", []) if isinstance(message, HumanMessage))
    triage = triage_request(latest)
    routing = classify_request(latest, state=state)
    # An answer to an interview question is not a new request. Preserve the
    # original scenario and report intent while merging the user's new facts.
    if state.get("needs_full_report") and not state.get("report_generated"):
        triage.update({
            "scenario": state.get("scenario", triage["scenario"]),
            "decision_goal": state.get("decision_goal", triage["decision_goal"]),
            "needs_full_report": True,
            "needs_interview": True,
            "complexity": state.get("complexity", triage["complexity"]),
            "hard_interview_limit": state.get("hard_interview_limit", triage["hard_interview_limit"]),
            "answer_type": "research_plan",
            "needs_clarification": False,
            "decision_template_id": state.get("decision_template_id", triage.get("decision_template_id", "")),
            "routing_confidence": state.get("routing_confidence", triage.get("routing_confidence", 0.0)),
        })
        routing.selected_skill = state.get("selected_skill") or routing.selected_skill
    # A direct-answer session may later become a new report-worthy request.
    # Do not inherit interview_complete/evidence/step counters from the old
    # narrow answer, otherwise the new scenario silently skips its interview.
    starting_new_research = bool(triage["needs_full_report"] and not state.get("needs_full_report", False))
    intent = identify_intent(all_user_text)
    # Triage is authoritative for whether this is a report-worthy request. Keep
    # the legacy intent for analyst compatibility, but expose the richer scenario.
    if not triage["needs_full_report"]:
        # Keep the legacy analyst intent (market_overview, entry_feasibility,
        # etc.) for compatibility; answer_type/scenario controls routing.
        intent = identify_intent(all_user_text)
    assumption_mode = stage in ("pre_launch", "planning") and intent != "roi_calculation"
    user_turn_count = state.get("user_turn_count", 1)
    interview_round = int(state.get("interview_round", 0))
    interview_complete = bool(state.get("interview_complete"))
    if starting_new_research:
        interview_round = 0
        interview_complete = False
    if triage["user_requested_start"] or not triage["needs_interview"]:
        interview_complete = True
    elif interview_round >= int(state.get("hard_interview_limit", triage["hard_interview_limit"])):
        interview_complete = True
    research_plan = build_research_plan(intent, stage)
    dimensions = state.get("evaluation_required_dimensions") or required_dimensions(triage["scenario"], triage["needs_full_report"])
    if dimensions:
        research_plan = [{"dimensions": dimensions, "mode": "scenario", "goal": triage["decision_goal"]}]
    # Only analyst/tool executions consume the Agent step budget. Supervisor
    # routing and human interview turns are tracked separately; counting them
    # here caused a six-question interview to exhaust MAX_AGENT_STEPS before
    # competition and business analysts could run.
    step = 0 if starting_new_research else state.get("agent_step_count", 0)
    status = ({dimension: "pending" for dimension in DIMENSIONS} if starting_new_research
              else dict(state.get("dimension_status", {dimension: "pending" for dimension in DIMENSIONS})))
    question_data: dict = {}
    question_bank = _scenario_question_bank(triage["scenario"], industry, region, budget) if triage["needs_interview"] else []
    interview_target = max(1, min(len(question_bank), triage["hard_interview_limit"] or len(question_bank)))
    readiness_score = min(100, round(interview_round / interview_target * 100)) if question_bank else 100
    consent_answered = "interview_consent" in set(state.get("covered_topics", []))

    if state.get("report_confirmation_answer") == "generate":
        question = ""
        decision, reason = "GENERATE_REPORT", "用户确认生成正式报告"
    elif triage.get("needs_clarification"):
        question_data = _question(
            "request_scope", "明确研究对象", "你希望研究哪个具体行业、公司、岗位或经营问题？",
            "先明确对象和要做的决定，才能避免联网搜索跑偏。",
            ["评估一个创业想法", "诊断现有业务", "研究某家公司", "回答一个具体问题"],
        )
        question = question_data["question"]
        decision, reason = "WAIT_USER", triage["routing_reason"]
    elif not triage["needs_full_report"]:
        question = ""
        decision, reason = "DIRECT_ANSWER", triage.get(
            "routing_reason", "问题范围已经明确，先直接联网回答，不启动完整报告流程"
        )
    elif industry == "unknown":
        facts = state.get("known_facts", {}) or {}
        scope = str(facts.get("industry", facts.get("request_scope", ""))).strip()
        if scope == "研究某家公司":
            question_data = _question(
                "company_name", "具体公司", "请直接写出要研究的公司名称，以及你最关心的业务、产品或技术。",
                "公司名称和关注范围决定搜索来源，避免把企业研究做成泛泛行业介绍。",
                ["例如：华为芯片设备", "例如：某公司的业务线", "例如：某公司的竞争力", "我先只看公司整体"],
            )
        elif scope in {"开店/本地生意", "互联网或AI产品", "制造业/工厂"} or "industry" in set(state.get("covered_topics", [])):
            question_data = _question(
                "industry_detail", "具体研究对象", "请具体写出行业、产品、设备或公司名称（例如：杭州咖啡店、AI客服产品、CNC机器人焊接产线）。",
                "上一轮只确定了大类；补充具体对象后，搜索和后续建议才会真正针对你的问题。",
                ["杭州咖啡店", "AI客服产品", "CNC机器人焊接产线", "宠物烘焙店"],
            )
        else:
            question_data = _question(
                "industry", "研究对象", "请直接写出你要调研的具体行业、生意、产品、设备或公司名称。",
                "先明确研究对象，后续联网搜索才不会跑偏。",
                ["杭州咖啡店", "AI教育产品", "CNC机器人焊接产线", "华为芯片设备"],
            )
        question = question_data["question"]
        decision, reason = "WAIT_USER", "研究对象还不够具体，先补充名称或产品范围"
    elif triage["needs_interview"] and not interview_complete and not consent_answered:
        question_data = _interview_consent_question(interview_target)
        question = question_data["question"]
        decision, reason = "WAIT_USER", "在开始研究前，请用户选择是否进入访谈模式"
    elif not interview_complete:
        interview_state = dict(state)
        if starting_new_research:
            interview_state["covered_topics"] = []
        question_data = build_scenario_question(interview_state, triage["scenario"], industry, region, budget, intent)
        if not question_data:
            interview_complete = True
            readiness_score = 100
            question = ""
            decision, reason = "CONTINUE_RESEARCH", "已收集足够背景或没有新的高价值问题，开始联网研究"
        else:
            question = question_data["question"]
            decision, reason = "WAIT_USER", "当前还有会改变建议方向的背景信息，先进行一次高价值访谈"
    elif state.get("awaiting_user"):
        question = state.get("pending_question", "请补充继续调研所需的信息。")
        decision, reason = "WAIT_USER", "缺少个性化测算参数，暂停等待用户补充"
    elif step >= MAX_AGENT_STEPS:
        question = ""
        if triage["needs_full_report"] and not state.get("report_generated"):
            decision, reason = "REPORT_CONFIRMATION", f"达到最大 Agent 步骤 {MAX_AGENT_STEPS}，研究阶段结束，等待报告确认"
        else:
            decision, reason = "FINISH", f"达到最大 Agent 步骤 {MAX_AGENT_STEPS}，完成当前任务"
    else:
        question = ""
        decision = "FINISH"
        for dimension in (dimensions or ROUTE_PRIORITY):
            if status.get(dimension, "pending") in ("pending", "researching"):
                decision = DIMENSION_TO_ANALYST[dimension]
                break
        if decision == "FINISH":
            if triage["needs_full_report"] and not state.get("report_generated"):
                decision = "REPORT_CONFIRMATION"
                reason = "研究阶段完成，等待用户确认是否生成正式报告"
            else:
                reason = "所有维度均已研究或明确标记证据不足，完成当前任务"
        else:
            reason = f"{dimension}维度待研究，调度{decision}"

    if decision == "CONTINUE_RESEARCH":
        decision = "FINISH"
        for dimension in (dimensions or ROUTE_PRIORITY):
            if status.get(dimension, "pending") in ("pending", "researching"):
                decision = DIMENSION_TO_ANALYST[dimension]
                reason = f"访谈信息已经足够，开始研究{dimension}维度"
                break
        if decision == "FINISH":
            decision = "REPORT_CONFIRMATION"
            reason = "没有待研究维度，等待用户确认是否生成正式报告"

    message = question or f"[Supervisor 决策] {reason}\n[下一节点] {decision}"
    result = {
        "messages": [AIMessage(content=message)],
        "industry": industry,
        "region": region,
        "budget": budget,
        "profile_confirmed": industry != "unknown",
        "interview_complete": interview_complete,
        "interview_round": interview_round + (
            1 if decision == "WAIT_USER" and question and question_data.get("topic") != "interview_consent" else 0
        ),
        "interview_target_questions": interview_target,
        "interview_readiness_score": readiness_score,
        "current_interview_question": question_data,
        "covered_topics": [] if starting_new_research else list(state.get("covered_topics", [])),
        "interview_questions": ([] if starting_new_research else list(state.get("interview_questions", []))) + ([question] if question else []),
        "user_intent": intent,
        "business_stage": stage,
        "assumption_mode": assumption_mode,
        "research_stage": "await_user" if decision == "WAIT_USER" else "research",
        "research_plan": research_plan,
        "next_agent": decision,
        "research_complete": decision in ("FINISH", "DIRECT_ANSWER", "REPORT_CONFIRMATION", "GENERATE_REPORT"),
        "awaiting_user": decision in ("WAIT_USER", "REPORT_CONFIRMATION"),
        "pending_question": question,
        "completion_reason": reason if decision == "FINISH" else state.get("completion_reason", ""),
        "supervisor_reason": reason,
        "agent_step_count": step,
        "scenario": triage["scenario"],
        "decision_goal": triage["decision_goal"],
        "decision_template_id": triage.get("decision_template_id", ""),
        "routing_confidence": triage.get("routing_confidence", 0.0),
        "answer_type": ("report_confirmation" if decision == "REPORT_CONFIRMATION" else triage["answer_type"] if decision != "WAIT_USER" else "clarifying_question"),
        "complexity": triage["complexity"],
        "hard_interview_limit": triage["hard_interview_limit"],
        "needs_web": triage["needs_web"],
        "needs_interview": triage["needs_interview"],
        "needs_full_report": triage["needs_full_report"],
        "user_requested_start": triage["user_requested_start"],
        "selected_skill": routing.selected_skill or "",
        "initial_skill": state.get("initial_skill") or routing.selected_skill or "",
        "skill_candidates": routing.candidates,
        "skill_confidence": routing.confidence,
        "skill_fallback_reason": routing.fallback_reason,
        "routing_classifier": routing.classifier,
        "skill_trace": list(state.get("skill_trace", [])) + [{
            "selected_skill": routing.selected_skill,
            "confidence": routing.confidence,
            "classifier": routing.classifier,
            "reason": routing.reason,
            "candidates": routing.candidates,
        }],
    }
    if starting_new_research:
        result.update({
            "evidence": [],
            "tool_events": [],
            "conflicts": [],
            "analyst_opinions": {},
            "scores": {dimension: 0 for dimension in DIMENSIONS},
            "report_generated": False,
            "report_confirmation_pending": False,
            "report_confirmation_answer": "",
            "citation_status": "pending",
            "citation_validation_error": "",
            "report_claims": [],
            "claim_judgments": [],
            "claim_entailment_rate": None,
            "claim_judge_status": "unavailable",
            "user_conditions": [],
            "recommendation_traces": [],
            "recommendation_trace_status": "pending",
            "completion_reason": "",
            "initial_skill": routing.selected_skill or "",
        })
    return result
