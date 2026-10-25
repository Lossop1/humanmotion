"""动作词表：能力测试与测试句子的共同来源。

提示词按 HumanML3D 训练文本里的高频写法确定（见 hm/mine_phrasings.py），
不按中文直译。约定：每条是一个可独立生成的单动作。
"""

# (键, 中文, 英文提示词, 时长帧)
ACTIONS_ALL = [
    # —— 位移 ——
    ("walk", "向前走", "a person walks forward", 52),
    ("walk_back", "向后退", "a person walks backward", 48),
    ("walk_left", "向左走", "a person walks to the left", 48),
    ("walk_right", "向右走", "a person walks to the right", 48),
    ("run", "跑", "a person runs forward", 40),
    ("jog", "慢跑", "a person jogs forward", 48),
    ("sprint", "冲刺", "a person sprints forward fast", 36),
    ("sidestep", "侧步", "a person sidesteps to the left", 44),
    ("climb", "上台阶", "a person climbs up stairs", 52),
    ("descend", "下台阶", "a person walks down stairs", 52),
    ("crawl", "爬行", "a person gets on all fours and crawls forward", 56),
    ("sneak", "蹑手蹑脚", "a person sneaks forward slowly", 48),
    ("skip", "蹦跳着走", "a person skips forward", 44),
    # —— 转向 ——
    ("turn", "转身", "a person turns around", 40),
    ("turn_left", "向左转", "a person walks forward and turns left", 48),
    ("turn_right", "向右转", "a person walks forward and turns right", 48),
    ("spin", "绕圈走", "a person walks in a circle", 48),
    # —— 站立与体位 ——
    ("stand", "站住", "a person stands still", 40),
    ("sit", "坐下", "a person sits down and stands back up", 52),
    ("stand_up", "站起来", "a person stands up from a chair", 44),
    ("kneel", "跪下", "a person kneels down on one knee", 44),
    ("crouch", "蹲下", "a person crouches down", 40),
    ("lie", "躺下", "a person lies down on the floor", 44),
    ("get_up", "起身", "a person gets up from the floor", 48),
    ("bend", "弯腰", "a person bends over forward", 40),
    ("stretch", "伸展", "a person stretches their arms above their head", 40),
    # —— 上肢 ——
    ("wave_hello", "挥手打招呼", "a person waves hello with the right hand", 40),
    ("wave_goodbye", "挥手告别", "a person waves goodbye", 40),
    ("clap", "鼓掌", "a person claps their hands", 40),
    ("raise_hands", "举手", "a person raises both hands", 36),
    ("point", "指向前方", "a person points forward with the right hand", 36),
    ("salute", "敬礼", "a person salutes with the right hand", 36),
    ("beckon", "招手示意", "a person beckons with the right hand", 36),
    ("hands_hips", "叉腰", "a person puts both hands on the hips", 36),
    ("cross_arms", "抱臂", "a person crosses both arms on the chest", 36),
    ("scratch_head", "挠头", "a person scratches their head with their right hand", 36),
    ("shake_head", "摇头", "a person shakes the head", 32),
    ("nod", "点头", "a person nods their head", 32),
    ("shrug", "耸肩", "a person shrugs the shoulders", 32),
    # —— 全身 ——
    ("jump", "向上跳", "a person jumps up in place", 36),
    ("jump_forward", "向前跳", "a person jumps forward once", 36),
    ("hop", "单脚跳", "a person hops on one leg", 36),
    ("kick", "踢腿", "a person kicks with their right leg", 36),
    ("punch", "出拳", "a person punches forward with the right hand", 36),
    ("throw", "投掷", "a person throws a ball forward", 36),
    ("push", "推", "a person pushes forward with both hands", 40),
    ("pull", "拉", "a person pulls backward with both hands", 40),
    ("pick", "捡东西", "a person bends down and picks something up", 52),
    ("dance", "跳舞", "a person dances", 60),
    ("squat", "深蹲", "a person squats down and stands up", 52),
]

KEYS = [a[0] for a in ACTIONS_ALL]
BY_KEY = {a[0]: a for a in ACTIONS_ALL}
