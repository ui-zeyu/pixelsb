"""User-visible Chinese copy."""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from pixelsb.domain import commands
from pixelsb.domain.classify import Classification
from pixelsb.domain.container import Block, BlockRole, ContainerReport, Finding
from pixelsb.domain.detect import Detection, detect_patterns
from pixelsb.domain.models import (
    BitChoice,
    BitsMask,
    CropMask,
    ExtractEncoding,
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
FILTER_PLACEHOLDER = "命令：b.0、r or g、thr 128、b > r and b > g、rect(0, 0, 10, 10)"
FILTER_ERROR = "命令错误："
FILTER_TIP = (
    "命令输入：一行写一个操作，第一个词决定它是哪一类。"
    "选中一条操作时框里就是它的命令，改完回车原地替换（种类也可以换）；"
    "框为空时输入会在最上面添加；点击配方空白处取消选中后框也是空的。"
    "打字过程中不预演，回车才应用；回车后框里写成这条命令的标准写法，与配方里那条一致。"
    "操作命令：thr 数值、xor 数值、inv、gray、crop，要单独一行。"
    "位选择：整行只有通道与位 token（b.0 只看 B 的 0 位、b 看整条通道、"
    "a 是 alpha 通道、all 是全部位），它们之间可用 or、and、not，如 r or g.0、all and not r。"
    "区域条件：其余文本（b > r、left < 10、rect(...)、B.0 == 1），"
    "条件之间照常用 and / or / not。"
    "and / or / not 只在同类之间使用：把位选择和区域条件写在一行会被拒绝，"
    "分成两行写，操作之间的组合交给配方栈。"
    "命令写在框正在编辑的那一条上（换种类也行）；框没有绑定操作时加到栈顶；"
    "位网格与快捷键只写位选择那一层，不会改写别类操作；落栈后可以用 ▲ ▼ 再调。"
    "数值写十进制（128）或带 0x 的十六进制（0x80），且不能超过这张图最宽通道的满值"
    "（8 位 255、16 位 65535）。"
    "区域字段：left、top、right、bottom 与各通道名（R、G、B…），"
    "像素占据 [left, right) × [top, bottom)；"
    "通道名是原始值，加 .bits 换成勾选位算出的值，加位号取该通道的某一位；"
    "rect(x0, y0, x1, y1) 选中矩形，grid(x, y, step_x, step_y) 按步长取点，参数均可具名。"
    "画布切到「选区」工具后拖动框选，回车把选区加为区域操作，Esc 取消预览。"
    "⌘F 聚焦，回车应用，Shift+回车在栈顶再叠一条（不动手里这条），"
    "Esc 先清空框里的文字、空框再按才删掉这一条并回到画布。"
)
DECIMAL = "十进制"
HEX = "十六进制"
BINARY = "二进制"
OPEN_FAILED = "无法打开图像"
FILTER_NO_MATCH = "过滤器未命中任何像素"
MODE_MOVE = "移动"
MODE_SELECT = "选区"
MODE_MOVE_TIP = "拖动画布平移视图"
MODE_SELECT_TIP = "拖动框选区域；回车把选区加为区域操作，Esc 取消"
THRESHOLD_LABEL = "阈值"
THRESHOLD_LEVEL_TIP = "阈值：0 到该图通道满值（8 位即 255），每个通道按 值 > 阈值 压成满值或 0"
XOR_LABEL = "异或"
XOR_VALUE_TIP = "异或值：把每个通道与这个常数按位异或，置位的位被翻转（16 位通道可填到 0xFFFF）"
VIEW_EXPORT = "导出"
VIEW_EXPORT_TIP = "把当前的位选择与值域操作的结果存为图像文件；区域操作的淡化是观看用的，不写进文件"
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
⌘C    复制读数"""

ORIGIN_LABEL = {
    SampleOrigin.RAW: "原始",
    SampleOrigin.PALETTE: "调色板",
    SampleOrigin.CONVERTED: "已转换",
}

SECTION_LAYERS = "配方"
LAYER_ADD = "＋"
LAYER_ADD_TIP = (
    "在最上面添加一个操作：越靠上的越晚生效。"
    "要参数的（位选择、区域、阈值、异或）直接在命令输入里输入。"
)
LAYER_UP = "▲"
LAYER_UP_TIP = "把选中的操作上移一条，让它更晚生效"
LAYER_DOWN = "▼"
LAYER_DOWN_TIP = "把选中的操作下移一条，让它更早生效"
LAYER_REMOVE = "✕"
LAYER_REMOVE_TIP = "删除选中的操作"
LAYER_ENABLED_TIP = "开关这一条；关掉的操作不参与画面和提取"
LAYER_BASE = "原图"
LAYER_BASE_NOTE = (
    "最底层：文件里的原始像素，始终生效，也不能删除。操作自下而上依次作用在它上面。"
    "在命令输入里输入即可添加，如 b.0、thr 128、xor 0xFF。"
)
LAYER_REGION_NOTE = "区域操作的表达式在命令输入里编辑：选中这一条，框里就是它。"
LAYER_BITS_NOTE = (
    "位网格在右侧「提取」栏：在那里勾选位，或用「原图」「全部最低位」预设；"
    "位选择也可以直接在命令输入里写，如 b.0、r or g。"
)
LAYER_COMMAND_NOTE = "参数在命令输入里改：选中这一条，框里就是它的命令，改完回车。"
BITS_PRESET_TIP = "把这条操作的位选择换成整幅原图，或换成每个通道的最低位"
LAYER_ALL_BITS = "全部位"
LAYER_NONE = "未选择"
EMPTY_EXPRESSION = "（空表达式，不过滤）"

MASK_BITS_TIP = (
    "位选择：画面画哪些位、提取流里打包哪些位。勾在下方网格里选。"
    "开了多层时只有最上面那一层算数：下面是它替换掉的位，可以关掉或删掉。"
)
MASK_BITS_SHADOWED = "上面还有开着的「位选择」操作：当前画面与提取用的是最上面那一条。"
MASK_REGION_TIP = "区域：按表达式筛掉像素，没通过的像素画布淡化、提取流跳过。"
MASK_CROP_TIP = "裁剪：把画布收缩到命中像素的范围，范围外不再显示。"
MASK_INVERT_TIP = "反相：每个通道按所在通道的最大值取反，翻转黑白二维码或负片式隐藏图案。"
MASK_GRAYSCALE_TIP = "灰度：三个颜色通道都换成像素亮度，弱化颜色干扰、凸显低对比度区域。"
MASK_THRESHOLD_TIP = (
    "阈值：每个通道按 值 > 阈值 压成满值或 0（阈值上限是该图通道的满值）；"
    "放在灰度之上就是黑白二值化。"
)
MASK_XOR_TIP = "异或：每个通道与常数按位异或，翻转常数置位的那些位；16 位通道填到 0xFFFF。"


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
}


def mask_info(mask: Mask) -> MaskInfo:
    """How a mask is named and described."""
    return MASK_INFO[type(mask)]


def mask_menu() -> tuple[Mask, ...]:
    """The masks the panel's add button offers: the ones that take no argument.

    A mask with a parameter of its own — the bits, the region expression, the
    threshold, the xor — is written in the filter box instead, where the command
    line carries the parameter the button cannot.
    """
    return (InvertMask(), GrayscaleMask(), CropMask())


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
