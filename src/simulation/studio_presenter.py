"""Chinese presentation of recorded V2 events; no inferred success or scene truth."""
SKILLS = {
    "OBSERVE_SCENE": ("观察场景", "拍摄 RGB-D 图像，更新物体位置"),
    "FIND_OBJECT": ("确认目标", "从视觉世界模型匹配目标物体"),
    "APPROACH_OBJECT": ("接近物体", "选择抓取姿态，避开观察到的障碍"),
    "GRASP_OBJECT": ("闭合夹爪", "用接触力控制夹紧程度"),
    "VERIFY_GRASP": ("检查抓牢", "检查双侧接触力和夹爪间隙"),
    "LIFT_OBJECT": ("抬起物体", "重新观察，检查物体是否跟随夹爪"),
    "MOVE_TO_TARGET": ("搬运物体", "分段规划路径，边走边观察"),
    "PLACE_OBJECT": ("放下物体", "下降、确认支撑、松开夹爪"),
    "RETREAT": ("机械臂退开", "离开物体，让相机重新看清"),
    "VERIFY_TASK": ("视觉验收", "再次拍照，检查位置关系与稳定性"),
    "SCAN_SCENE": ("多视角扫描", "更换观察角度，补充视觉证据"),
    "MOVE_CAMERA": ("调整相机", "移动观察相机，寻找更清晰的视野"),
}
COLORS = dict(red="红色", blue="蓝色", green="绿色", yellow="黄色", cyan="青色", purple="紫色", orange="橙色", white="白色", gray="灰色", pink="粉色", black="黑色")
CATEGORIES = dict(cup="杯子", box="盒子", plate="盘子", bottle="瓶子", obstacle="障碍物", object="物体", region="空桌面区域")
STATES = dict(IDLE="等待任务", UNDERSTAND="理解指令", OBSERVE="观察环境", PLAN="制定计划", EXECUTE="执行技能", MONITOR="检查执行结果", VERIFY="核验结果", RECOVER="处理异常", REPLAN="重新规划", SUCCESS="任务完成", FAILED="任务未完成")
FAILURES = dict(TARGET_NOT_FOUND="未找到目标", LOW_CONFIDENCE="识别把握不足", DEPTH_INVALID="深度数据无效", IK_FAILED="当前姿态无法到达", PATH_BLOCKED="路径被阻挡", COLLISION_RISK="检测到碰撞风险", NO_CONTACT="未检测到有效夹持", GRASP_FAILED="抓取失败", OBJECT_SLIPPED="物体可能滑脱", TARGET_LOST="暂时看不到目标", PLACEMENT_FAILED="放置未通过检查", RELATION_NOT_SATISFIED="最终位置关系不符合要求", USER_STOP="已收到停止指令", RUNTIME_ERROR="执行或渲染服务异常", ValueError="无法理解任务，请说明要移动的物体和目标位置", LookupError="目标描述不明确，请补充颜色或类别", TimeoutError="任务超过时间限制")
ACTIONS = dict(REOBSERVE="重新观察", CHANGE_GRASP="更换抓取方式", CHANGE_APPROACH="更换接近方向", REPLAN="重新规划路径", ABORT="停止尝试")
PURPOSES = dict(interpret="理解指令", plan="安排技能顺序", recover="分析失败原因", visual_detection="分析图像")
PURPOSES.update(visual_grounding='从相机图像寻找目标',grasp_strategy='选择已验证的抓取候选')
FAILURES.update(NO_VALID_PLACEMENT='没有满足约束的安全放置位置',AMBIGUOUS_REFERENCE_FRAME='物体朝向不明确，无法确定它的正前方',
                AMBIGUOUS_TARGET='已发现相关物体，但尚不能确认你指的是哪一个')
PURPOSES.update(visual_disambiguation='比较高清候选图像')
PURPOSES.update(visual_comparison='联合比较候选整体外形')


def object_name(obj):
    return COLORS.get(obj.get("color"), obj.get("color") or "") + CATEGORIES.get(obj.get("category"), obj.get("category") or "物体")


def failure_text(data):
    evidence=data.get('evidence',{})
    if evidence.get('spatial_rejection')=='NO_VALID_PLACEMENT: SUPPORT_EXTENT':
        return '已找到物体，但测得的支撑面不足以稳定放下它'
    if evidence.get('reason')=='force_limit':
        return '夹爪接触力超过安全上限，已停止动作'
    if evidence.get('reason_code')=='MODEL_CONTENT_REJECTED':
        return '模型服务端的图片内容审核拒绝了请求；需向服务提供方反馈核查'
    if evidence.get('reason_code')=='MODEL_REQUEST_FAILED':
        status=evidence.get('http_status')
        return f'视觉模型请求失败（HTTP {status}），请查看接口错误事件' if status else '视觉模型服务暂时不可用'
    if evidence.get('identity_review_required'):
        return '目标身份待复核，尚未授权继续操作'
    if evidence.get('reason_code')=='IDENTITY_BUDGET_EXHAUSTED':
        return '该目标的视觉确认次数已用完，仍未确认，已停止操作；已确认的物体保持不变'
    return FAILURES.get(data.get('failure'),data.get('failure','执行中出现错误'))


def human_event(event):
    if event['type']=='identity_joint_comparison':
        d=event['data']
        return ('info','联合比较已选出候选，等待新图确认身份与位置。') if d.get('selected') else (
            'warning','候选联合比较仍有歧义或结构矛盾，需要补充证据。')
    if event['type']=='identity_side_observation':
        if event['data']['status']=='unavailable':
            return 'warning','未取得这个候选的清楚侧视图：安全路径或可见程度不足。'
        return 'info',('已拍摄新的高清侧面图，等待模型复核。' if event['data']['status']=='captured'
            else '外形尚不明确，正在规划安全的侧面观察位置。')
    if event['type']=='identity_evidence_rejected':
        d=event['data']
        if d.get('show_in_studio') is False:return None
        refs='、'.join(d.get('task_references',[]))
        retained=d.get('previous_image_retained',False)
        return 'info',(f'寻找「{refs}」时：' if refs else '')+('新照片被遮挡或深度不足，保留之前的清楚照片。' if retained
            else '新照片被遮挡或深度不足，暂不用于候选判断。')
    if event['type']=='identity_review_inconsistent':
        return 'warning','模型的外形/颜色检查与最终判断不一致，暂不确认这个候选。'
    if event['type']=='identity_candidate_review':
        d=event['data'];verdict={'yes':'符合','no':'不符合','uncertain':'需要补充观察'}.get(d['review']['verdict'],'待确认')
        return 'info',f"独立核对候选 {d['candidate_id']}：{verdict}。"
    if event['type']=='identity_candidates':
        return ('info',f"已整理 {len(event['data']['candidate_ids'])} 个候选，交给视觉模型比较。")
    if event['type']=='identity_comparison_batch':
        return ('info','已完成一批候选核对，继续确认最终目标。')
    if event['type']=='identity_comparison':
        return ('info' if event['data'].get('selected') else 'warning','已对比候选图像，等待所选物体的新鲜定位。' if event['data'].get('selected') else '候选比较无法唯一确定目标，停止操作并等待补充观察。')
    if event['type']=='identity_corrected':
        return ('warning','模型对比新旧候选后纠正了原身份绑定，重新测量位置。')
    if event['type']=='identity_review_required':
        return ('warning',event['data']['query']+'：出现身份关联冲突，操作前需模型重新确认。')
    if event['type']=='identity_state':
        return None
    if event['type']=='observation_motion':
        return None
    if event['type']=='visual_identity':
        d=event['data'];names={'confirmed':'模型已确认目标身份','not_visible':'当前视野未看到目标，继续换角度','ambiguous':'发现多个疑似目标，需要进一步区分','detail_needed':'细节不足，切换高清图像','occluded':'目标被遮挡，需要改变视角','attribute_conflict':'模型所指区域与指令颜色冲突，补充观察','instance_conflict':'不同目标指向同一物体，需要重新区分','invalid_box':'模型图像坐标无效，需要重新确认','depth_unreliable':'身份已识别，深度几何仍不足','model_error':'视觉模型请求失败，有限重试','model_unavailable':'视觉模型不可用，禁止本地代替确认','budget_exhausted':'该目标的视觉确认次数已用完，停止操作'}
        return ('info' if d['status']=='confirmed' else 'warning',f"{names.get(d['status'],d['status'])}：{d['query']}")
    if event['type']=='identity_revoked':
        return ('warning','目标身份跟踪失效，需重新看图确认：'+event['data']['query'])

    """Return (tone, text) only when there is a meaningful recorded event."""
    kind, d = event["type"], event["data"]
    if kind=='stage':
        stages={'APPROACHING_FOR_OBSERVATION':'接近目标，准备近距离观察。','LOCAL_REOBSERVATION':'重新拍摄腕部图像，确认位置。',
            'PRECISE_LOCKED':'已取得精定位，继续检查身份与抓取条件。','POST_SCAN_IDENTITY_REVIEW':'扫描后发现身份疑问，先由模型确认再规划抓取。','NEED_MORE_GEOMETRY':'现有几何信息不足，准备换角度观察。',
            'MULTIVIEW_SCAN':'移动腕部相机，补充侧面形状。','PLAN_PLACEMENT_REGION':'按空间关系和厘米间距计算可用位置。',
            'PUSH_CORRECTION':'执行小步推移，随后重新测量。','REGRASP':'重新抓取并调整放置位置。'}
        return ('info',stages[d['stage']]) if d.get('stage') in stages else None
    if kind=='target_candidates':
        if d.get('reason')=='category_conflict':
            return 'info',f"已发现可能是「{d['query']}」的物体，正在核对图像与内部分类。"
        return 'info',f"「{d['query']}」有多个候选，正在对比图像。"
    if kind=='detail_capture':
        return 'info',f"已拍摄 1280×960 高清图像，核对 {len(d['candidate_ids'])} 个候选物体。"
    if kind=='target_binding':
        category=d.get('provisional_category')
        mismatch=category and d.get('requested_category') and category!=d['requested_category']
        note=f"；内部暂记为{CATEGORIES.get(category,category)}" if mismatch else ''
        method='高清复核确认' if d.get('source')=='high_resolution_visual_review' else '当前图像找到'
        return 'success',f"{method}「{d['query']}」对应的物体{note}，后续持续跟踪该物体。"
    if kind=='visual_grounding_retry':
        return 'warning',f"「{d['query']}」的定位框超出相机画面，正在用标准化坐标重新识别。"
    if kind=='visual_disambiguation':
        if d.get('selected'):return 'success',f"高清复核确认目标：{d['selected']}。"
        return 'warning',f"高清复核后仍无法确定目标：{d.get('reason','候选之间缺少足够的视觉区别')}。"
    if kind=='adaptive_grasp':
        path='简单物体快速抓取' if d['fast_path_used'] else '多视角几何抓取' if d['multiview_used'] else '局部几何抓取'
        return 'info',f"{path} · {'侧向' if d['strategy']=='side_grasp' else '顶部'}接近 · {'调用了模型选择候选' if d['llm_grasp_reasoner_called'] else '本地选择候选'}。"
    if kind=='placement_correction':
        names={'PUSH':'小步推移','REGRASP':'重新抓取','REPLACE':'保持夹持并重新放置','ABORT':'停止修正'}
        return 'warning',f"位置尚未满足要求，修正方式：{names.get(d['strategy'],d['strategy'])}。"
    if kind=='relation_validation':
        rows=[f"要求 {r['requested_gap_m']*100:.1f} cm，实测 {r['actual_gap_m']*100:.1f} cm" for r in d.get('constraints',[]) if r.get('requested_gap_m') is not None and r.get('actual_gap_m') is not None]
        return ('success' if d.get('accepted') else 'warning','；'.join(rows)) if rows else None
    if kind=='push_observation':return 'info',f"推移后重新拍摄：测得位移 {sum(x*x for x in d['observed_delta_m'][:2])**.5*100:.1f} cm。"
    if kind == "state":
        state, skill = d["state"], d.get("skill")
        if state in {"EXECUTE", "VERIFY"} and skill:
            title, detail = SKILLS.get(skill, (skill, ""))
            return "info", f"{title} · {detail}"
        if state == "MONITOR": return None
        if state == "SUCCESS": return "success", "新图像验收通过：物体已释放，位置关系和稳定性符合要求。"
        if state == "FAILED": return "error", failure_text(d) + "。可查看原始事件定位原因。"
        return "info", {"UNDERSTAND":"正在理解你的指令。", "OBSERVE":"正在从真实相机图像建立世界模型。", "PLAN":"正在根据目标和已观察到的环境制定计划。", "RECOVER":"执行遇到问题，正在检查证据。", "REPLAN":"调整后继续执行。"}.get(state, STATES.get(state,state))
    if kind == "llm_request": return "info", f"请求 {d['model']}：{PURPOSES.get(d['purpose'],d['purpose'])}。"
    if kind == "llm": return "success", f"{d['model']} 已返回{PURPOSES.get(d['purpose'],d['purpose'])}结果，通过结构检查。"
    if kind == "llm_error": return "warning", f"模型请求失败：{d['error_type']}。"
    if kind in {"llm_fallback", "interpret_backend"}: return "warning", "本次使用本地规则解析；这一步没有使用 LLM。"
    if kind == "interpretation":
        target = object_name(d['target']) if d.get('target') else '可见的空桌面'
        return "info", f"任务目标：{object_name(d['object'])} → {target}。"
    if kind == "plan": return "info", f"已生成 {len(d['steps'])} 个技能步骤；列表随真实执行更新。"
    if kind == "multiview_fusion": return "info", f"已融合 {d['views']} 个视角，获得 {d['objects']} 组物体观测。"
    if kind == "grasp_candidates": return "info", f"根据视觉几何生成 {len(d)} 个抓取候选，开始检查可达性。"
    if kind == "grasp_validation": return "info", f"夹持测量：左侧 {d['left_force']:.2f} N，右侧 {d['right_force']:.2f} N，夹爪间隙 {d['width']*1000:.1f} mm。"
    if kind == "grasp_stability":
        return ("success" if d['stable'] else "warning",
                f"闭爪稳定检查：{d['bilateral_samples']}/3 次保持双侧接触，"
                f"爪口变化 {d['jaw_width_drift_m']*1000:.1f} mm。")
    if kind == "probe_lift_validation":
        return "info", f"小幅试提：物体上升 {d['object_motion'][2]*1000:.1f} mm，正核对是否跟随夹爪。"
    if kind == "placement_plan_reused": return "info", "目标再次确认后，沿用搬运前已验证的放置位置。"
    if kind == "lift_validation": return "info", f"视觉测得物体抬升 {d['object_motion'][2]*1000:.1f} mm；正在核对跟随运动。"
    if kind == 'geometric_validation':
        return 'info','已取得位置与稳定性测量，等待身份复核及最终验收。'
    if kind == "final_validation":
        if 'stable_displacement_m' in d:
            return "info", f"验收测量：两帧之间位移 {d['stable_displacement_m']*1000:.2f} mm；释放状态：{'已释放' if d['object_released'] else '仍被夹持'}。"
        return "warning", "视觉验收证据不足，尚不能判定完成。"
    if kind == "failure": return "warning", f"{SKILLS.get(d['skill'],(d['skill'],))[0]}遇到问题：{failure_text(d)}。"
    if kind == "recovery": return "warning", f"恢复策略：{ACTIONS.get(d.get('action'),d.get('action','停止'))}。"
    return None
