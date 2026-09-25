"""User-visible Chinese copy."""

from decimal import ROUND_HALF_UP, Decimal

from pixelsb.domain.classify import Classification
from pixelsb.domain.container import Block, BlockRole, ContainerReport, Finding
from pixelsb.domain.detect import Detection, detect_patterns
from pixelsb.domain.models import BitChoice, ExtractEncoding, LoadedImage, SampleOrigin, ViewerState
from pixelsb.domain.readout import build_readout, label_zoom

_TWO_PLACES = Decimal("0.01")

APP_NAME = "pixelsb"
FILE_MENU = "文件"
VIEW_MENU = "视图"
HELP_MENU = "帮助"
OPEN = "打开…"
QUIT = "退出"
FILE_INFO = "文件信息"
PANEL_BITS = "▦"
PANEL_BITS_TIP = "位选择与提取"
PANEL_INFO = "ⓘ"
PANEL_INFO_TIP = "文件信息"
SHORTCUTS = "快捷键"
ORIGINAL = "原图"
ALL_LSB = "全部最低位"
ZOOM_IN = "+"
ZOOM_OUT = "−"
ZOOM_FIT = "适配"
ZOOM_RESET = "1:1"
SECTION_BITS = "位选择"
SECTION_EXTRACT = "提取"
CHANNEL_TIP = "勾选整条通道"
COLUMN_TIP = "勾选整列"
PLANE_PREV = "◀"
PLANE_NEXT = "▶"
PLANE_PREV_TIP = "上一个位平面（按 R7…R0、G7… 的顺序反向切换，到头循环）"
PLANE_NEXT_TIP = "下一个位平面（按 R7…R0、G7… 的顺序切换，到头循环）"
CHANNEL_PREV = "▲"
CHANNEL_NEXT = "▼"
CHANNEL_PREV_TIP = "上一个通道的整条通道（R → G → B 循环）"
CHANNEL_NEXT_TIP = "下一个通道的整条通道（R → G → B 循环）"
EXTRACT_SEARCH_TIP = "搜索十六进制或 ASCII"
EXTRACT_OFFSET = "偏移"
EXTRACT_ENCODING_TIP = "点击切换右侧文本的解读编码：ASCII、UTF-8、UTF-16LE、UTF-16BE"
ORDER_CHANNEL_TIP = "通道顺序：每个像素里先读哪条通道的位（StegSolve 的 RGB、BGR 等排列）"
ORDER_BIT_MSB_TIP = "位序：先出现的位放进字节最高位；整条 8 位通道全选时得到按位反转的字节"
ORDER_BIT_LSB_TIP = "位序：先出现的位放进字节最低位；整条 8 位通道全选时得到的正是原始字节值"
ORDER_SCAN_XY_TIP = "扫描顺序：先横后纵，一行走完再换下一行"
ORDER_SCAN_YZ_TIP = "扫描顺序：先纵后横，一列走完再换下一列"
EXTRACT_ENCODINGS: tuple[tuple[ExtractEncoding, str], ...] = (
    (ExtractEncoding.ASCII, "ASCII"),
    (ExtractEncoding.UTF8, "UTF-8"),
    (ExtractEncoding.UTF16_LE, "UTF-16LE"),
    (ExtractEncoding.UTF16_BE, "UTF-16BE"),
)
EXTRACT_ENCODING_LABELS: dict[ExtractEncoding, str] = dict(EXTRACT_ENCODINGS)
FILTER_PLACEHOLDER = "显示过滤器：rect(50, 50, 100, 100) and B >= R"
FILTER_ERROR = "过滤错误："
FILTER_TIP = (
    "显示过滤器同时作用于画布和提取。字段：left、top、right、bottom 与各通道名（R、G、B…）；"
    "像素占据 [left, right) × [top, bottom)，即 right = left + 1、bottom = top + 1。"
    "通道名就是通道原始值；加 .bits 换成勾选位算出的值，加位号取该通道的某一位，"
    "如 R.0 是最低位、R.7 是最高位（16 位通道到 .15）。"
    "rect 选中矩形，四条边可具名：rect(left=50, top=50, right=100, bottom=100)，"
    "也可按顺序写 rect(50, 50, 100, 100)，与四个字段的比较完全等价。"
    "grid(x, y, step_x, step_y) 从 (x, y) 开始，横向、纵向各按步长取点："
    "grid(10, 20, 6, 6) 每 6 个像素取 1 个，步长 1 即逐像素，四项均可具名。"
    "画布切到「选区」工具后拖动框选：拖动与松开时只是预览（画布保留选区框、状态栏显示范围与命中数），"
    "回车把 (原表达式) and rect 追加进过滤器，Esc 取消预览。"
    "例：rect(50, 50, 100, 100) and B >= R、grid(10, 20, 6, 6) and R.0 == 1。"
    "⌘F 聚焦，回车应用，Esc 清空并回到画布，留空即不过滤。"
)
DECIMAL = "十进制"
HEX = "十六进制"
BINARY = "二进制"
OPEN_FAILED = "无法打开图像"
ONLY_MATCHED = "仅提取像素"
ONLY_MATCHED_TIP = "勾选后画布收缩为过滤器命中的像素，未命中的行列被移除；取消勾选恢复淡化显示"
FILTER_NO_MATCH = "过滤器未命中任何像素"
MODE_MOVE = "移动"
MODE_SELECT = "选区"
MODE_MOVE_TIP = "拖动画布平移视图"
MODE_SELECT_TIP = "拖动框选区域；回车把选区填入过滤器，Esc 取消"
ADJUST_INVERT = "反相"
ADJUST_GRAYSCALE = "灰度"
ADJUST_THRESHOLD = "阈值"
ADJUST_INVERT_TIP = "视图反相：每个通道取 255 − 原值，翻转黑白二维码或负片式隐藏图案"
ADJUST_GRAYSCALE_TIP = "视图转灰度：按亮度合成通道，弱化颜色干扰、凸显低对比度区域"
ADJUST_THRESHOLD_TIP = "视图二值化：通道值大于阈值的置白，其余置黑（在灰度之后、反相之前生效）"
THRESHOLD_LEVEL_TIP = "阈值：0–255，画布按 通道值 > 阈值 二值化"
VIEW_EXPORT = "导出"
VIEW_EXPORT_TIP = "把当前位组合画面（含反相、灰度、阈值）存为图像文件"
EXTRACT_SAVE = "保存"
EXTRACT_SAVE_TIP = "把整条提取字节流写入文件"
FRAME_PREV = "◀"
FRAME_NEXT = "▶"
FRAME_PREV_TIP = "上一帧"
FRAME_NEXT_TIP = "下一帧"
INFO_TITLE = "文件信息"
SECTION_SCAN = "检查"
SECTION_EXIF = "EXIF"
INFO_NO_EXIF = "无 EXIF 信息"
WARNING_MARK = "⚠"
BLOCK_RENDER_TIP = "选中把这一块按所属的数据流渲染成像素"
DUMP_TIP = "点击在下方查看这一块的十六进制转储（含块头与校验）"
RENDER_FAILED = "无法渲染所选块"
ANY_FILE = "所有文件 (*)"
SAVE_FAILED = "无法保存"
DETECTION_TIP = "点击跳到提取流中的这一段"


def open_failed(detail: str) -> str:
    return f"{OPEN_FAILED}：{detail}"


def save_failed(detail: str) -> str:
    return f"{SAVE_FAILED}：{detail}"


def saved_to(path: str) -> str:
    return f"已保存 {path}"


def more_detections(count: int) -> str:
    return f"还有 {count} 处…"


def filter_count(passed: int, total: int) -> str:
    return f"通过 {passed}/{total}"


def selection_status(x0: int, y0: int, x1: int, y1: int) -> str:
    """Status-bar line while dragging the region preview."""
    return f"选区 rect({x0}, {y0}, {x1 + 1}, {y1 + 1})  {x1 - x0 + 1}×{y1 - y0 + 1} 像素"


def selection_ready(x0: int, y0: int, x1: int, y1: int, count: str = "") -> str:
    """Status-bar line for a settled preview; ``count`` is a prebuilt pass label."""
    hits = f"{count} · " if count else ""
    return f"{selection_status(x0, y0, x1, y1)} · {hits}回车追加到过滤器，Esc 取消"


def zoom_label(zoom: float) -> str:
    """The zoom with the times sign, at most two decimals, rounded half up."""
    value = Decimal(zoom).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
    return f"{value:.2f}".rstrip("0").rstrip(".") + "×"


NO_IMAGE = "未打开图像"
NO_CURSOR = "移动鼠标或方向键查看像素"
CANVAS_HINT = "打开一张图像，或把文件拖到这里"
CANVAS_SHORTCUT = "⌘O 选择文件"
CONVERTED_NOTE = "位平面来自转换后的数据"
IMAGE_FILTER = (
    "图像 (*.png *.bmp *.gif *.tif *.tiff *.webp *.jpg *.jpeg *.ppm *.pgm *.pbm);;所有文件 (*)"
)
ZOOM_TIP = "拖动缩放。⌘滚轮、触控板捏合和 +、- 也可以"
ZOOM_RESET_TIP = "按原始像素大小显示（1 倍）"
FORMAT_TIP = "F 在十进制、十六进制、二进制之间切换"
MATRIX_TIP = (
    "勾选位来组合画面和数字。行首、列首的复选框选整行或整列。⌘、Ctrl 或 Shift 加单击＝只看这一位。"
)
SHORTCUT_HELP = """⌘O    打开
方向键    移动光标 1 像素
Shift+方向键    移动 8 像素
左键拖动    平移
选区工具    拖动框选（预览）；回车追加 rect 过滤，Esc 取消；按住空格临时平移
单击位    勾选或取消这一位
行首 / 列首复选框    选整行或整列
⌘/Shift+单击    只看这一位
空格拖拽、中键拖拽    平移
⌘滚轮 / ⌘+双指滑动 / 双指捏合    缩放
0    适配窗口
1:1    原始像素大小
[ / ]    当前通道的上一位 / 下一位
R G A L    画面只看该通道的最低位
1–9    按顺序只看该通道的最低位
F    切换进制
⌘C    复制读数"""

ORIGIN_LABEL = {
    SampleOrigin.RAW: "原始",
    SampleOrigin.PALETTE: "调色板",
    SampleOrigin.CONVERTED: "已转换",
}


LAYER_ALL = "原图"
LAYER_NONE = "未选择"


def layer_text(bits: tuple[BitChoice, ...] | None) -> str:
    """Which bits drive the canvas: every bit, none, or the list by name."""
    if bits is None:
        return LAYER_ALL
    if not bits:
        return LAYER_NONE
    return " ".join(f"{choice.plane}{choice.bit}" for choice in bits)


def readout_text(state: ViewerState) -> str:
    if state.image is None:
        return NO_IMAGE
    readout = build_readout(state)
    if readout is None:
        return NO_CURSOR
    lines = [f"光标 ({readout.cursor.x}, {readout.cursor.y})"]
    lines.extend(f"{channel.name}  {channel.text}" for channel in readout.channels)
    lines.append(f"画面 {layer_text(readout.bits)}")
    return "\n".join(lines)


def status_info(state: ViewerState) -> str:
    """Status bar, left: what is open. The frame cluster owns the frame position."""
    image = state.image
    if image is None:
        return NO_IMAGE
    note = f"  {CONVERTED_NOTE}" if image.any_converted else ""
    return (
        f"{image.path.name}  {image.width}×{image.height}  模式 {image.source_mode}  "
        f"{planes_text(image)}{note}"
    )


def frame_status(index: int, count: int, delay_ms: int) -> str:
    """The status-bar frame indicator: the position, plus the frame's own delay."""
    position = f"帧 {index + 1}/{count}"
    return f"{position} · {delay_ms}ms" if delay_ms else position


def classification_note(classification: Classification | None) -> str:
    """How the extract panel names the stream, empty when nothing was recognized."""
    if classification is None:
        return ""
    return f"类型 {classification.mime} · {classification.engine}"


def finding_lines(report: ContainerReport) -> list[str]:
    """One line per anomaly the census flagged, for the amber warning block."""
    return [
        _FINDING_TEXT[finding.kind].format(offset=finding.offset, length=finding.length)
        for finding in report.findings
    ]


def payload_lines(findings: tuple[Finding, ...]) -> list[str]:
    """Signature and flag hits inside the suspicious payloads, offsets relative."""
    lines: list[str] = []
    for finding in findings:
        lines.extend(
            _hit_text(detection, f"+0x{detection.offset:x}")
            for detection in detect_patterns(finding.payload)
        )
        if finding.decoded:
            lines.extend(
                _hit_text(detection, "解压后") for detection in detect_patterns(finding.decoded)
            )
    return lines


def _hit_text(detection: Detection, where: str) -> str:
    """One hit line: a flag form gets quoted, a file signature gets named."""
    if detection.flagged:
        return f"疑似 flag「{detection.label}」（{where}）"
    return f"内嵌 {detection.label} 文件签名（{where}）"


def block_role_text(block: Block) -> str:
    """The census's verdict for one block, plus its stream when it has one."""
    role = "必需" if block.role is BlockRole.REQUIRED else "附属"
    return f"{role} · 流 {block.group}" if block.group else role


def canvas_note(block: Block, count: int) -> str:
    """Which blocks the canvas is showing: a numbered stream, or one block.

    Opening a file shows its first stream, so the note a fresh page carries is
    the same sentence clicking that stream's name produces.
    """
    if not block.group:
        return f"画布：{block.label}（按扫描行渲染）"
    members = f"{block.label} 等 {count} 个块" if count > 1 else block.label
    return f"画布：流 {block.group}（{members}）"


def original_note() -> str:
    """What the canvas shows when the census found no stream to name."""
    return "画布：原图"


def block_preview(block: Block) -> str:
    """The block's payload as printable text — a comment reads, pixels do not."""
    return block.preview


def more_blocks(count: int) -> str:
    return f"… 其余 {count} 个块从略"


def dump_caption(block: Block, guessed: str = "") -> str:
    """The dump's caption: which block, where it sits, how long, and its type."""
    end = block.offset + block.length
    guess = f" · 推测 {guessed}" if guessed else ""
    return f"块转储 · {block.label} · 0x{block.offset:08x} – 0x{end:08x} · {block.length} B{guess}"


def render_failed(detail: str) -> str:
    return f"{RENDER_FAILED}：{detail}"


_FINDING_TEXT = {
    "duplicate-eof": "发现重复的结束标记（CVE-2023-28303 截图残留或手工拼接的痕迹）",
    "idat-extra": "IDAT 数据流结束之后还有 {length} 字节不会被渲染（起于 0x{offset:x}）",
    "idat-gap": "IDAT 块不连续：0x{offset:x} 处夹有其他块，其后的数据可能被阅读器忽略",
    "idat-oversize": "像素数据解压后比图像需要多 {length} 字节，多出的扫描行不会被显示",
    "stream-truncated": "0x{offset:x} 起的图像数据流不完整",
}


def file_facts(image: LoadedImage) -> str:
    """The file card's format line: size, mode, frames."""
    frames = f"帧 {image.frame_count}" if image.frame_count > 1 else "单帧"
    return f"{image.width}×{image.height} · 模式 {image.source_mode} · {frames}"


def planes_text(image: LoadedImage) -> str:
    """The channel line: each plane's name, depth, and origin."""
    return ", ".join(
        f"{plane.name} {plane.bit_depth}bit {ORIGIN_LABEL[plane.origin]}" for plane in image.planes
    )


def status_view(state: ViewerState) -> str:
    """Status bar, right: where the cursor is and what zoom would show numbers."""
    if state.image is None:
        return ""
    location = "—" if state.cursor is None else f"({state.cursor.x}, {state.cursor.y})"
    needed = label_zoom(state)
    if needed is not None and state.zoom < needed:
        return f"光标 {location}  放到 {needed}× 显示数值"
    return f"光标 {location}"
