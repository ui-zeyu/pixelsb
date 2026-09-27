"""User-visible Chinese copy."""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from pixelsb.domain import commands
from pixelsb.domain.classify import Classification
from pixelsb.domain.container import Block, BlockRole, ContainerReport, Finding, SizeHint
from pixelsb.domain.detect import Detection, detect_patterns
from pixelsb.domain.models import (
    ArnoldMask,
    BitChoice,
    BitsMask,
    CropMask,
    ExtractEncoding,
    ExtractOrder,
    FftMask,
    FrameGeometry,
    GrayscaleMask,
    InvertMask,
    LoadedImage,
    Mask,
    Raster,
    RegionMask,
    SampleOrigin,
    SamplePlane,
    ThresholdMask,
    ViewerState,
    XorMask,
)
from pixelsb.domain.readout import build_readout, label_zoom
from pixelsb.domain.scan import ScanHit
from pixelsb.domain.selection import whole

_TWO_PLACES = Decimal("0.01")

APP_NAME = "pixelsb"
FILE_MENU = "文件"
VIEW_MENU = "视图"
HELP_MENU = "帮助"
OPEN = "打开…"
QUIT = "退出"
FILE_INFO = "文件信息"
PANEL_EXTRACT = "▦"
PANEL_EXTRACT_TIP = "提取"
PANEL_INFO = "ⓘ"
PANEL_INFO_TIP = "文件信息"
PANEL_SCAN = "⌕"
PANEL_SCAN_TIP = "扫描"
PANEL_ARNOLD = "猫"
PANEL_ARNOLD_TIP = "猫脸变换爆破"
PANEL_HISTOGRAM = "▥"
PANEL_HISTOGRAM_TIP = "直方图与卡方"
SHORTCUTS = "快捷键"
COMMAND_HELP_TITLE = "命令语法"
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
PLANE_PREV_TIP = "上一位平面"
PLANE_NEXT_TIP = "下一位平面"
CHANNEL_PREV = "▲"
CHANNEL_NEXT = "▼"
CHANNEL_PREV_TIP = "上一条通道"
CHANNEL_NEXT_TIP = "下一条通道"
EXTRACT_SEARCH_TIP = "搜索十六进制或 ASCII"
EXTRACT_OFFSET = "偏移"
EXTRACT_ENCODING_TIP = "切换右列文本的编码"
ORDER_CHANNEL_TIP = "像素内先读哪条通道的位"
ORDER_BIT_MSB_TIP = "先出现的位放在字节高位"
ORDER_BIT_LSB_TIP = "先出现的位放在字节低位"
ORDER_SCAN_XY_TIP = "先横后纵"
ORDER_SCAN_YZ_TIP = "先纵后横"
EXTRACT_ENCODINGS: tuple[tuple[ExtractEncoding, str], ...] = (
    (ExtractEncoding.ASCII, "ASCII"),
    (ExtractEncoding.UTF8, "UTF-8"),
    (ExtractEncoding.UTF16_LE, "UTF-16LE"),
    (ExtractEncoding.UTF16_BE, "UTF-16BE"),
)
EXTRACT_ENCODING_LABELS: dict[ExtractEncoding, str] = dict(EXTRACT_ENCODINGS)
FILTER_PLACEHOLDER = "命令：b.0、r or g、thr 128、b > r and b > g、rect(0, 0, 10, 10)"
FILTER_ERROR = "命令错误："
FILTER_TIP = "回车应用 · ⇧回车另叠一条 · Esc 清空，空框再按删除这一条"
COMMAND_HELP = """\
命令输入：一行写一个操作，第一个词决定它是哪一类。
选中一条操作时框里就是它的命令，改完回车原地替换（种类也可以换）；框为空时输入会在最上面添加。
打字过程中不预演，回车才应用；回车后框里写成这条命令的标准写法，与配方里那条一致。

操作命令（单独一行，不与别的内容组合）
  thr 数值          值 > 阈值 压成满值或 0
  xor 数值          每个通道与常数按位异或
  inv               每个通道按满值取反
  gray              三个颜色通道换成亮度
  crop              画布收缩到命中像素的范围
  fft               位选择勾到的通道换成对数幅度谱
  arnold 次数 a b   猫映射逆变换重排像素（与常见脚本同名同序），要方图

位选择（整行只有通道与位）
  b      整条通道        b.0    只看这一位       all    全部位
  or / and / not   并、交、补，如 r or g.0、all and not r

区域条件（其余文本）
  left top right bottom     像素占据 [left, right) × [top, bottom)
  通道名                    原始值；加 .bits 是勾选位的值，加位号（R.3）是那一位
  rect(x0, y0, x1, y1)      矩形，参数可具名 left= top= right= bottom=
  grid(x, y, step_x, step_y)    从起点按步长取点
  比较、算术、and / or / not 照常组合，如 b > r and b > g

写法
  数值：十进制 128，或 0x 十六进制；不能超过这张图最宽通道的满值（8 位 255、16 位 65535）
  位选择和区域条件不能写在一行：分成两行，操作之间的组合由配方栈负责
  命令写在框正在编辑的那一条上；框没有绑定操作时加到栈顶
  位网格与快捷键只写位选择那一层，不会改写别类操作

按键
  ⌘F 聚焦命令框    回车应用    Shift+回车在栈顶再叠一条
  Esc 先清空框里的文字；空框再按才删掉这一条并回到画布
"""
DECIMAL = "十进制"
HEX = "十六进制"
BINARY = "二进制"
OPEN_FAILED = "无法打开图像"
FILTER_NO_MATCH = "过滤器未命中任何像素"
MODE_MOVE = "移动"
MODE_SELECT = "选区"
MODE_MOVE_TIP = "拖动画布平移视图"
MODE_SELECT_TIP = "拖动框选；回车加为区域操作，Esc 取消"
THRESHOLD_LEVEL_TIP = "每个通道按 值 > 阈值 压成满值或 0"
XOR_VALUE_TIP = "每个通道与这个常数按位异或"
VIEW_EXPORT = "导出"
VIEW_EXPORT_TIP = "当前画面存为图像文件（区域淡化只用于显示）"
EXTRACT_SAVE = "保存"
EXTRACT_SAVE_TIP = "把整条提取字节流写入文件"
FRAME_PREV = "◀"
FRAME_NEXT = "▶"
FRAME_PREV_TIP = "上一帧"
FRAME_NEXT_TIP = "下一帧"
INFO_TITLE = "文件信息"
SECTION_SCAN = "检查"
SECTION_FRAMES = "帧"
SECTION_EXIF = "EXIF"
INFO_NO_EXIF = "无 EXIF 信息"
FRAME_HEADERS = ("帧", "尺寸", "偏移", "延时")
SIZE_HINT_NOTE = "声明的宽高放不下这些像素，可修补为："
SIZE_OPEN_TIP = "按此宽高改写 IHDR 并打开修补后的文件"
WARNING_MARK = "⚠"
BLOCK_RENDER_TIP = "把这一块所在的流渲染到画布"
DUMP_TIP = "查看这一块的十六进制转储"
RENDER_FAILED = "无法渲染所选块"
ANY_FILE = "所有文件 (*)"
SAVE_FAILED = "无法保存"
DETECTION_TIP = "点击跳到提取流中的这一段"


_LONG_TEXT = 64  # past this, file-derived text gets zero-width breaks so it can wrap
_BREAK_AFTER = "/\\._-"
_BREAK_EVERY = 24


def breakable(text: str) -> str:
    """File-derived text with zero-width break points, so a long path or name wraps
    inside the panel instead of dragging a scrollbar in.

    Separators are the natural cuts, and a run without any (a hex dump saved as a
    name) is cut every so often anyway; short text is left exactly as it was.
    """
    if len(text) <= _LONG_TEXT:
        return text
    parts: list[str] = []
    run = 0
    for char in text:
        parts.append(char)
        run += 1
        if char in _BREAK_AFTER or run >= _BREAK_EVERY:
            parts.append("\u200b")
            run = 0
    return "".join(parts)


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
    return f"{selection_status(x0, y0, x1, y1)} · {hits}回车加为区域操作，Esc 取消"


def zoom_label(zoom: float) -> str:
    """The zoom with the times sign, at most two decimals, rounded half up."""
    value = Decimal(zoom).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
    return f"{value:.2f}".rstrip("0").rstrip(".") + "×"


NO_IMAGE = "未打开图像"
NO_CURSOR = "移动鼠标或方向键查看像素"
NOT_IN_VIEW = "该像素不在当前视图中"
CANVAS_HINT = "打开一张图像，或把文件拖到这里"
CANVAS_SHORTCUT = "⌘O 选择文件"
CONVERTED_NOTE = "位平面来自转换后的数据"
IMAGE_FILTER = (
    "图像 (*.png *.bmp *.gif *.tif *.tiff *.webp *.jpg *.jpeg *.ppm *.pgm *.pbm);;所有文件 (*)"
)
ZOOM_TIP = "⌘滚轮、捏合或 +、− 缩放"
ZOOM_RESET_TIP = "按原始像素大小显示（1 倍）"
FORMAT_TIP = "F 在十进制、十六进制、二进制之间切换"
MATRIX_TIP = "单击勾选；⌘、Ctrl 或 Shift 加单击只看这一位"

SHORTCUT_HELP = """⌘O    打开
方向键    移动光标 1 像素
Shift+方向键    移动 8 像素
左键拖动    平移
选区工具    拖动框选（预览）；回车把选区加为区域操作，Esc 取消；按住空格临时平移
单击位    勾选或取消这一位（右侧「提取」栏的位网格里）
行首 / 列首复选框    选整行或整列
⌘/Shift+单击    只看这一位
空格拖拽、中键拖拽    平移
⌘滚轮 / ⌘+双指滑动 / 双指捏合    缩放
0    适配窗口
1:1    原始像素大小
[ / ]    当前通道的上一位 / 下一位
R G B A L    画面只看该通道的最低位
1–9    按顺序只看该通道的最低位
F    切换进制
⌘F    命令输入：b.0、thr 128、xor 0xFF、区域表达式
⌘1–⌘5    切换右侧栏页面：提取 / 扫描 / 猫 / 直方图 / 文件信息
⌘C    复制读数"""

ORIGIN_LABEL = {
    SampleOrigin.RAW: "原始",
    SampleOrigin.PALETTE: "调色板",
    SampleOrigin.CONVERTED: "已转换",
}

SECTION_LAYERS = "配方"
LAYER_ADD = "＋"
LAYER_ADD_TIP = "添加操作；越靠上越晚生效，要参数的写进命令框"
LAYER_UP = "▲"
LAYER_UP_TIP = "上移（更晚生效）"
LAYER_DOWN = "▼"
LAYER_DOWN_TIP = "下移（更早生效）"
LAYER_REMOVE = "✕"
LAYER_REMOVE_TIP = "删除这一条"
LAYER_ENABLED_TIP = "开关这一条"
LAYER_BASE = "原图"
BITS_PRESET_TIP = "换成整幅原图，或每通道的最低位"
LAYER_ALL_BITS = "全部位"
LAYER_NONE = "未选择"
EMPTY_EXPRESSION = "（空表达式，不过滤）"

MASK_BITS_TIP = "勾选画面与提取流用哪些位"
MASK_BITS_SHADOWED = "上面还有开着的「位选择」操作：当前画面与提取用的是最上面那一条。"
MASK_REGION_TIP = "按表达式筛掉像素，没通过的淡化且不进提取流"
MASK_CROP_TIP = "把画布收缩到命中像素的范围"
MASK_INVERT_TIP = "每个通道按满值取反"
MASK_GRAYSCALE_TIP = "三个颜色通道换成像素亮度"
MASK_THRESHOLD_TIP = "每个通道按 值 > 阈值 压成满值或 0"
MASK_XOR_TIP = "每个通道与常数按位异或"
MASK_FFT_TIP = "勾到的每个通道换成对数幅度谱（直流居中）"
MASK_ARNOLD_TIP = "猫映射逆变换重排像素；arnold 次数 a b，要方图"


@dataclass(frozen=True, slots=True)
class MaskInfo:
    """How one kind of mask is named and described."""

    label: str
    tip: str


MASK_INFO: dict[type[Mask], MaskInfo] = {
    BitsMask: MaskInfo("位选择", MASK_BITS_TIP),
    RegionMask: MaskInfo("区域", MASK_REGION_TIP),
    CropMask: MaskInfo("裁剪", MASK_CROP_TIP),
    InvertMask: MaskInfo("反相", MASK_INVERT_TIP),
    GrayscaleMask: MaskInfo("灰度", MASK_GRAYSCALE_TIP),
    ThresholdMask: MaskInfo("阈值", MASK_THRESHOLD_TIP),
    XorMask: MaskInfo("异或", MASK_XOR_TIP),
    FftMask: MaskInfo("频谱", MASK_FFT_TIP),
    ArnoldMask: MaskInfo("猫脸变换", MASK_ARNOLD_TIP),
}


def mask_info(mask: Mask) -> MaskInfo:
    """How a mask is named and described."""
    return MASK_INFO[type(mask)]


def mask_menu() -> tuple[Mask, ...]:
    """The masks the panel's add button offers: the ones that take no argument.

    A mask with a parameter of its own — the bits, the region expression, the
    threshold, the xor, the cat map — is written in the filter box instead, where
    the command line carries the parameter the button cannot.
    """
    return (InvertMask(), GrayscaleMask(), CropMask(), FftMask())


def mask_detail(mask: Mask, planes: tuple[SamplePlane, ...]) -> str:
    """A mask's parameters as the command line that means it, for the layer list."""
    match mask:
        case RegionMask(expression=expression):
            return expression.strip() or EMPTY_EXPRESSION
        case BitsMask(selection=selection):
            if whole(planes, selection):
                return bits_text(planes, selection)
            return commands.text_of(mask, planes) or bits_text(planes, selection)
        case _:
            return commands.text_of(mask, planes) or ""


def bits_text(planes: tuple[SamplePlane, ...], selection: frozenset[BitChoice]) -> str:
    """The two words for the selections no command spells: every bit, or none."""
    return LAYER_ALL_BITS if whole(planes, selection) else LAYER_NONE


def layer_text(bits: tuple[BitChoice, ...] | None) -> str:
    """Which bits drive the canvas: every bit, none, or the list by name.

    Every bit is ``全部位`` rather than the base layer's own name: value masks can
    leave the selection whole while the numbers behind it are not the file's.
    """
    if bits is None:
        return LAYER_ALL_BITS
    if not bits:
        return LAYER_NONE
    return " ".join(f"{choice.plane}{choice.bit}" for choice in bits)


def readout_text(state: ViewerState, raster: Raster | None) -> str:
    if state.image is None:
        return NO_IMAGE
    if state.cursor is None:
        return NO_CURSOR
    readout = None if raster is None else build_readout(state, raster)
    if readout is None:
        return NOT_IN_VIEW
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


def more_blocks(count: int) -> str:
    return f"… 其余 {count} 个块从略"


def size_hint_label(hint: SizeHint) -> str:
    return f"{hint.width}×{hint.height}"


def frame_row(geometry: FrameGeometry) -> tuple[str, str, str, str]:
    """One GIF frame's table line: position in the animation, place on the canvas."""
    return (
        str(geometry.index + 1),
        f"{geometry.width}×{geometry.height}",
        f"+{geometry.x}+{geometry.y}",
        f"{geometry.delay} ms",
    )


def more_frames(count: int) -> str:
    return f"… 其余 {count} 帧从略"


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


def status_view(state: ViewerState, raster: Raster | None) -> str:
    """Status bar, right: where the cursor is and what zoom would show numbers."""
    if state.image is None:
        return ""
    location = "—" if state.cursor is None else f"({state.cursor.x}, {state.cursor.y})"
    needed = None if raster is None else label_zoom(state, raster)
    if needed is not None and state.zoom < needed:
        return f"光标 {location}  放到 {needed}× 显示数值"
    return f"光标 {location}"


# --- the scan page and the cat-map gallery ----------------------------------

SECTION_SWEEP = "扫描"
SECTION_ARNOLD = "猫脸变换"
SECTION_HISTOGRAM = "直方图"
SECTION_CHI2 = "卡方"
HISTOGRAM_TIP = "读画布所示的值：位选择只留勾中的位，区域淡化不剔除像素"
CHI2_TIP = "p 接近 1：这一位的值对像被交换过，即该位像被写入过"
SWEEP_START = "开始扫描"
SWEEP_STOP = "停止"
SWEEP_COLUMNS = ("命令", "顺序", "预览", "结果")
SWEEP_NOTHING = "—"


def sweep_candidates(total: int) -> str:
    return f"{total} 个候选" if total else "无可扫描候选"


def sweep_progress(done: int, total: int) -> str:
    return f"{done}/{total}"


def sweep_done(flagged: int, total: int) -> str:
    hits = f"命中 {flagged}" if flagged else "无命中"
    return f"{hits} · {total} 个候选"


def sweep_order(order: ExtractOrder) -> str:
    """The candidate's read order: bit end first, then which axis runs fastest."""
    return f"{order.bit_order.value.upper()} · {order.scan.value.upper()}"


def sweep_result(hit: ScanHit) -> str:
    """The stream's verdict: its type and up to two distinct flags, or a dash."""
    parts = [hit.label] if hit.label else []
    flags = list(dict.fromkeys(hit.flags))
    parts.extend(flags[:2])
    if len(flags) > 2:
        parts.append(f"另有 {len(flags) - 2} 个")
    return " · ".join(parts) if parts else SWEEP_NOTHING


ARNOLD_NEEDS_SQUARE = "需要方图：当前 {width}×{height}"
ARNOLD_TIMES = "次数"
ARNOLD_A = "a"
ARNOLD_B = "b"
RANGE_DASH = "–"
ARNOLD_START = "开始"
ARNOLD_STOP = "停止"
ARNOLD_CANCELLED = "已停止 · {count} 个候选"


def arnold_not_square(width: int, height: int) -> str:
    return ARNOLD_NEEDS_SQUARE.format(width=width, height=height)


def arnold_pick_tip(times: int, a: int, b: int, score: float) -> str:
    """The thumbnail's tooltip: the command it writes, and the heuristic's say."""
    return f"arnold {times} {a} {b}\n平滑度 {score:.2f}（越大越像正常图片）"


def arnold_caption(times: int, a: int, b: int) -> str:
    return f"{times} {a} {b}"


def arnold_progress(done: int, total: int) -> str:
    return f"尝试中 {done}/{total}"


def arnold_cancelled(count: int) -> str:
    return ARNOLD_CANCELLED.format(count=count)


def arnold_done(count: int) -> str:
    return f"{count} 个候选"


def chi2_detail(plane: str, bit: int, statistic: float, degrees: int) -> str:
    """One cell's tooltip: the pair test's numbers behind the p it shows."""
    return f"{plane} 位 {bit} · χ² {statistic:.0f} · 自由度 {degrees}"
