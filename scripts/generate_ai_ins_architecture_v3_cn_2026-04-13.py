from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math

W, H = 1600, 900
OUT = '/root/.openclaw/workspace/2026-04-13-22-18-00-ai-ins-multi-agent-a2a-architecture-v3-cn.png'
OUT2 = '/root/.openclaw/workspace/2026-04-13-22-18-00-ai-ins-multi-agent-a2a-architecture-v3-cn@2x.png'
FONT_REG = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def vertical_gradient(size, top, bottom):
    w, h = size
    img = Image.new('RGBA', size)
    px = img.load()
    for y in range(h):
        t = y / max(h - 1, 1)
        c = lerp(top, bottom, t)
        for x in range(w):
            px[x, y] = c
    return img


def rounded_mask(size, radius):
    m = Image.new('L', size, 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle((0, 0, size[0]-1, size[1]-1), radius=radius, fill=255)
    return m


def paste_round_grad(base, box, top, bottom, radius, border=None, border_width=2):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    grad = vertical_gradient((w, h), top, bottom)
    mask = rounded_mask((w, h), radius)
    base.paste(grad, (x1, y1), mask)
    if border:
        d = ImageDraw.Draw(base)
        d.rounded_rectangle(box, radius=radius, outline=border, width=border_width)


def shadow_layer(size, box, radius, fill=(40, 74, 110, 48), blur=14, offset=(0, 8)):
    lay = Image.new('RGBA', size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    x1, y1, x2, y2 = box
    ox, oy = offset
    d.rounded_rectangle((x1+ox, y1+oy, x2+ox, y2+oy), radius=radius, fill=fill)
    return lay.filter(ImageFilter.GaussianBlur(blur))


def add_shadow(base, box, radius, fill=(40, 74, 110, 48), blur=14, offset=(0, 8)):
    base.alpha_composite(shadow_layer(base.size, box, radius, fill, blur, offset))


def pill(draw, box, fill, text, fg='white', outline=None, size=24, bold=True):
    draw.rounded_rectangle(box, radius=(box[3]-box[1])//2, fill=fill, outline=outline, width=2 if outline else 1)
    f = font(size, bold=bold)
    bbox = draw.textbbox((0, 0), text, font=f)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x = (box[0]+box[2]-tw)/2
    y = (box[1]+box[3]-th)/2 - 2
    draw.text((x, y), text, font=f, fill=fg)


def fit_text(draw, text, box, max_size, min_size, bold=False, fill=(24, 42, 58), align='center', line_spacing=6):
    x1, y1, x2, y2 = box
    best = None
    for sz in range(max_size, min_size-1, -1):
        f = font(sz, bold=bold)
        bbox = draw.multiline_textbbox((0, 0), text, font=f, spacing=line_spacing, align=align)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        if tw <= (x2-x1) and th <= (y2-y1):
            best = (f, tw, th)
            break
    if not best:
        f = font(min_size, bold=bold)
        bbox = draw.multiline_textbbox((0, 0), text, font=f, spacing=line_spacing, align=align)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    else:
        f, tw, th = best
    if align == 'center':
        tx = x1 + (x2-x1-tw)/2
    else:
        tx = x1
    ty = y1 + (y2-y1-th)/2
    draw.multiline_text((tx, ty), text, font=f, fill=fill, spacing=line_spacing, align=align)


def card(base, box, title_cn, title_en, accent, fill=(255,255,255,235), title_fill=(30,52,72), en_fill=(102,118,132), radius=20):
    add_shadow(base, box, radius, fill=(accent[0], accent[1], accent[2], 42), blur=10, offset=(0, 6))
    d = ImageDraw.Draw(base)
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=(accent[0], accent[1], accent[2], 80), width=2)
    d.rounded_rectangle((box[0], box[1], box[0]+10, box[3]), radius=radius, fill=accent)
    title_box = (box[0]+24, box[1]+12, box[2]-18, box[1]+52)
    en_box = (box[0]+24, box[1]+50, box[2]-18, box[3]-12)
    fit_text(d, title_cn, title_box, 24, 18, bold=True, fill=title_fill, align='left', line_spacing=4)
    fit_text(d, title_en, en_box, 15, 11, bold=False, fill=en_fill, align='left', line_spacing=4)


def group_container(base, box, title, subtitle, color_top, color_bottom, badge_text=None):
    add_shadow(base, box, 28, fill=(60, 90, 130, 36), blur=18, offset=(0, 10))
    paste_round_grad(base, box, color_top, color_bottom, 28, border=(255,255,255,160), border_width=2)
    overlay = Image.new('RGBA', base.size, (0,0,0,0))
    od = ImageDraw.Draw(overlay)
    # soft inner tint
    od.rounded_rectangle((box[0]+8, box[1]+8, box[2]-8, box[3]-8), radius=22, outline=(255,255,255,86), width=1)
    base.alpha_composite(overlay)
    d = ImageDraw.Draw(base)
    right_limit = box[2]-18 if not badge_text else box[2]-168
    header_box = (box[0]+18, box[1]+14, right_limit, box[1]+46)
    fit_text(d, title, header_box, 28, 20, bold=True, fill=(27, 51, 79), align='left')
    sub_box = (box[0]+18, box[1]+46, right_limit, box[1]+74)
    fit_text(d, subtitle, sub_box, 15, 11, bold=False, fill=(86,105,124), align='left')
    if badge_text:
        pill(d, (box[2]-150, box[1]+14, box[2]-24, box[1]+44), fill=(255,255,255,210), text=badge_text, fg=(42,72,110), size=17, bold=True)


def draw_arrow(draw, p1, p2, color, width=7, head=16, both=False):
    draw.line((p1, p2), fill=color, width=width)
    def head_at(a, b):
        ang = math.atan2(b[1]-a[1], b[0]-a[0])
        left = (b[0] - head*math.cos(ang - math.pi/7), b[1] - head*math.sin(ang - math.pi/7))
        right = (b[0] - head*math.cos(ang + math.pi/7), b[1] - head*math.sin(ang + math.pi/7))
        draw.polygon([b, left, right], fill=color)
    head_at(p1, p2)
    if both:
        head_at(p2, p1)


def arrow_label(draw, text, center, fill, bg=(255,255,255,230)):
    f = font(16, bold=True)
    bbox = draw.textbbox((0,0), text, font=f)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    box = (center[0]-tw//2-10, center[1]-th//2-6, center[0]+tw//2+10, center[1]+th//2+6)
    draw.rounded_rectangle(box, radius=14, fill=bg, outline=(fill[0], fill[1], fill[2], 110), width=1)
    draw.text((box[0]+10, box[1]+5), text, font=f, fill=fill)


img = Image.new('RGBA', (W, H), (248, 251, 255, 255))
bg = vertical_gradient((W, H), (246, 250, 255, 255), (255, 255, 255, 255))
img.alpha_composite(bg)
d = ImageDraw.Draw(img)

# background decorative waves
for i, alpha in enumerate([40, 28, 20]):
    layer = Image.new('RGBA', (W, H), (0,0,0,0))
    ld = ImageDraw.Draw(layer)
    y = 110 + i*44
    ld.rounded_rectangle((80+i*26, y, 720+i*48, y+170), radius=80, fill=(110, 154, 230, alpha))
    ld.rounded_rectangle((870-i*30, 480-i*24, 1500-i*20, 720-i*20), radius=90, fill=(74, 202, 189, alpha))
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(42)))

# Title
fit_text(d, 'AI + 惯导多智能体 A2A 协同架构图', (56, 32, 980, 86), 38, 26, bold=True, fill=(20, 46, 76), align='left')
fit_text(d, '中文优化版 · 分层更清楚、配色更有区分、适合汇报/论文方案示意', (58, 82, 980, 108), 18, 12, bold=False, fill=(97,115,132), align='left')
pill(d, (1290, 34, 1540, 72), fill=(28, 90, 150), text='v3 中文增强版', fg='white', size=18, bold=True)

# Main blocks
knowledge_box = (56, 130, 450, 390)
orchestrator_box = (500, 138, 1098, 290)
a2a_box = (560, 396, 1040, 806)
left_box = (48, 398, 490, 826)
right_box = (1110, 398, 1552, 826)

group_container(img, knowledge_box, '知识与策略增强层', 'Knowledge / Evaluation / Optimization', (238, 244, 255, 240), (225, 236, 255, 248), 'LLM 增强')
group_container(img, left_box, 'INS-A 本地确定性执行栈', 'Deterministic Local Stack', (232, 249, 246, 244), (219, 242, 239, 250), '实时层')
group_container(img, right_box, 'INS-B 本地确定性执行栈', 'Deterministic Local Stack', (232, 249, 246, 244), (219, 242, 239, 250), '实时层')
group_container(img, a2a_box, 'A2A 协同层', 'Multi-INS Collaboration / Redundancy / Coordination', (229, 248, 251, 244), (213, 242, 248, 250), '分布式')

# Orchestrator block
add_shadow(img, orchestrator_box, 34, fill=(16, 82, 142, 58), blur=18, offset=(0, 10))
paste_round_grad(img, orchestrator_box, (28, 97, 166, 255), (23, 131, 176, 255), 34, border=(255,255,255,160), border_width=2)
d = ImageDraw.Draw(img)
pill(d, (736, 148, 864, 184), fill=(255,255,255,220), text='主控层', fg=(25, 91, 145), size=18, bold=True)
fit_text(d, '主控编排智能体', (580, 188, 1018, 226), 38, 28, bold=True, fill=(255,255,255), align='center')
fit_text(d, 'Orchestrator', (650, 228, 950, 254), 18, 14, bold=False, fill=(222, 241, 255), align='center')
fit_text(d, '任务拆解 · 流程编排 · 人工确认点 · 异常回滚', (570, 254, 1030, 278), 18, 12, bold=False, fill=(220,242,255), align='center')

# Knowledge cards
purple = (118, 108, 218)
blue = (72, 128, 219)
cyan = (34, 160, 184)
knowledge_cards = [
    ((168, 214, 338, 266), '数字孪生', 'Digital Twin', (87, 120, 210)),
    ((78, 274, 248, 326), '知识库检索', 'Knowledge-RAG', purple),
    ((258, 274, 428, 326), '文献调研', 'DeepResearch', blue),
    ((78, 334, 248, 386), '评估诊断', 'Evaluation', cyan),
    ((258, 334, 428, 386), '路径寻优', 'Policy Optimizer', (67, 136, 224)),
]
for box, cn, en, accent in knowledge_cards:
    card(img, box, cn, en, accent)

# Left stack cards
left_cards = [
    ((70, 476, 264, 558), '电机执行', 'Motion Control', (50, 166, 152)),
    ((274, 476, 468, 558), '数据采集', 'Data Acquisition', (59, 170, 146)),
    ((70, 572, 264, 654), '环境补偿', 'Environment Compensation', (39, 160, 153)),
    ((274, 572, 468, 654), '滤波解算', 'Filter Solver', (40, 132, 180)),
    ((70, 668, 264, 750), '健康监测', 'Health Monitor', (73, 157, 189)),
    ((274, 668, 468, 750), '安全监护', 'Safety Guard', (26, 129, 161)),
]
for box, cn, en, accent in left_cards:
    card(img, box, cn, en, accent)

# Right stack cards
right_cards = [
    ((1132, 476, 1326, 558), '电机执行', 'Motion Control', (50, 166, 152)),
    ((1336, 476, 1530, 558), '数据采集', 'Data Acquisition', (59, 170, 146)),
    ((1132, 572, 1326, 654), '环境补偿', 'Environment Compensation', (39, 160, 153)),
    ((1336, 572, 1530, 654), '滤波解算', 'Filter Solver', (40, 132, 180)),
    ((1132, 668, 1326, 750), '健康监测', 'Health Monitor', (73, 157, 189)),
    ((1336, 668, 1530, 750), '安全监护', 'Safety Guard', (26, 129, 161)),
]
for box, cn, en, accent in right_cards:
    card(img, box, cn, en, accent)

# A2A cards
mid_cards = [
    ((590, 476, 820, 560), '时间同步 / 参考对齐', 'Time Sync & Reference Alignment', (36, 170, 190)),
    ((780, 476, 1010, 560), '互校准', 'Cross-Calibration', (42, 154, 190)),
    ((590, 584, 820, 668), '一致性融合', 'Consensus Fusion', (58, 148, 194)),
    ((780, 584, 1010, 668), 'FDIR 故障检测隔离恢复', 'Fault Detection, Isolation & Recovery', (43, 133, 182)),
    ((686, 692, 914, 776), '角色切换 / 降级模式', 'Role Switching & Degraded Mode', (34, 146, 169)),
]
for box, cn, en, accent in mid_cards:
    card(img, box, cn, en, accent)

# Connection arrows
navy = (39, 98, 153)
teal = (39, 143, 160)
green = (46, 154, 143)
# knowledge -> orchestrator
p1 = (438, 232)
p2 = (500, 214)
draw_arrow(d, p1, p2, navy, width=8, head=18)
arrow_label(d, '知识 / 策略输入', (472, 184), navy)
# orchestrator down
p1 = (800, 290)
p2 = (800, 396)
draw_arrow(d, p1, p2, navy, width=8, head=18)
arrow_label(d, '任务编排 / 状态汇总', (942, 340), navy)
# left -> a2a
p1 = (490, 610)
p2 = (560, 610)
draw_arrow(d, p1, p2, teal, width=8, head=18, both=True)
arrow_label(d, '本地状态 / 约束', (525, 570), teal)
# right -> a2a
p1 = (1110, 610)
p2 = (1040, 610)
draw_arrow(d, p1, p2, teal, width=8, head=18, both=True)
arrow_label(d, '融合 / 协同反馈', (1072, 570), teal)
# orchestrator to local stacks
p1 = (650, 290)
p2 = (316, 398)
draw_arrow(d, p1, p2, green, width=6, head=15)
p1 = (950, 290)
p2 = (1288, 398)
draw_arrow(d, p1, p2, green, width=6, head=15)

# Legend / footer
legend_box = (58, 836, 1542, 882)
add_shadow(img, legend_box, 22, fill=(43, 74, 104, 30), blur=10, offset=(0, 5))
d.rounded_rectangle(legend_box, radius=22, fill=(255,255,255,220), outline=(205,218,232), width=2)
pill(d, (84, 846, 196, 874), fill=(33, 105, 167), text='LLM 规划层', fg='white', size=15, bold=True)
pill(d, (212, 846, 344, 874), fill=(42, 149, 161), text='A2A 协同层', fg='white', size=15, bold=True)
pill(d, (360, 846, 504, 874), fill=(46, 157, 142), text='确定性执行层', fg='white', size=15, bold=True)
fit_text(d, '核心原则：LLM 负责规划、解释、调度、诊断与知识增强；实时控制、滤波解算与安全联锁必须由确定性程序承担。', (530, 842, 1518, 878), 19, 12, bold=False, fill=(52,72,92), align='left')

# save
img.save(OUT)
img.resize((W*2, H*2), Image.Resampling.LANCZOS).save(OUT2)
print(OUT)
print(OUT2)
