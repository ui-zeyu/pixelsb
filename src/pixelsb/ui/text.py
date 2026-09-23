"""User-visible Chinese copy."""

from pixelsb.domain.models import SampleOrigin, ViewerState
from pixelsb.domain.readout import build_readout, label_zoom

APP_NAME = "pixelsb"
FILE_MENU = "文件"
VIEW_MENU = "视图"
HELP_MENU = "帮助"
OPEN = "打开…"
QUIT = "退出"
SHORTCUTS = "快捷键"
ORIGINAL = "原图"
ALL_LSB = "全部最低位"
ZOOM_IN = "+"
ZOOM_OUT = "−"
ZOOM_FIT = "适配"
ZOOM_RESET = "1:1"
DETACH = "分离"
CANVAS_LAYER = "画面"
NUMBER_LAYER = "数字"
READOUT_RESET = "原始值"
CHANNEL_TIP = "勾选整条通道"
COLUMN_TIP = "勾选整列"
EXTRACT_NOTE = "提取（按通道顺序、位从低到高，每 8 位拼 1 字节，高位在前）"
EXTRACT_SEARCH_TIP = "搜索十六进制或 ASCII"
FILTER_PLACEHOLDER = "显示过滤器：B >= R and B >= G"
FILTER_ERROR = "过滤错误："
FILTER_TIP = (
    "显示过滤器同时作用于画布和提取。字段：left、top（别名 x、y）与各通道名。"
    "例：50 <= left <= 100 and 50 <= top <= 100、B >= R and B >= G。"
    "⌘F 聚焦，留空即不过滤。"
)
VALUE = "数值"
DECIMAL = "十进制"
HEX = "十六进制"
BINARY = "二进制"
ABSOLUTE = "绝对值"
OFFSET = "相对锚点"
CLEAR_ANCHOR = "清除锚点"
ANCHOR = "锚点"
OPEN_FAILED = "无法打开图像"


def open_failed(detail: str) -> str:
    return f"{OPEN_FAILED}：{detail}"


NO_IMAGE = "未打开图像"
NO_CURSOR = "移动鼠标或方向键查看像素"
CANVAS_HINT = "打开一张图像，或把文件拖到这里"
CONVERTED_NOTE = "位平面来自转换后的数据"
IMAGE_FILTER = (
    "图像 (*.png *.bmp *.gif *.tif *.tiff *.webp *.jpg *.jpeg *.ppm *.pgm *.pbm);;所有文件 (*)"
)
ZOOM_TIP = "拖动缩放。⌘滚轮和 +、- 也可以"
ZOOM_RESET_TIP = "按原始像素大小显示（1 倍）"
DETACH_TIP = "勾选后画面与数字各自勾选；默认同步"
READOUT_RESET_TIP = "数字层回到原始通道值"
FORMAT_TIP = "F 在十进制、十六进制、二进制之间切换"
VALUE_MODE_TIP = "O 切换绝对值与相对锚点"
MATRIX_TIP = (
    "勾选位来组合画面和数字。勾选「分离」后，画面格与数字格各自控制，"
    "数字默认显示原始通道值。行首、列首的复选框选整行或整列。"
    "⌘、Ctrl 或 Shift 加单击＝只看这一位。"
)
SHORTCUT_HELP = """⌘O    打开
方向键    移动光标 1 像素
Shift+方向键    移动 8 像素
单击图像    设置锚点
左键拖动    平移
单击位    勾选或取消这一位
行首 / 列首复选框    选整行或整列
⌘/Shift+单击    只看这一位
分离    画面与数字各自勾选
Esc    清除锚点
空格拖拽、中键拖拽    平移
⌘滚轮、+、-    缩放
0    适配窗口
1:1    原始像素大小
[ / ]    当前通道的上一位 / 下一位
R G A L    画面只看该通道的最低位
1–9    按顺序只看该通道的最低位
F    切换进制
O    绝对值 / 相对锚点
⌘C    复制读数"""

ORIGIN_LABEL = {
    SampleOrigin.RAW: "原始",
    SampleOrigin.PALETTE: "调色板",
    SampleOrigin.CONVERTED: "已转换",
}


def readout_text(state: ViewerState) -> str:
    if state.image is None:
        return NO_IMAGE
    readout = build_readout(state)
    if readout is None:
        return NO_CURSOR
    lines = [f"光标 ({readout.cursor.x}, {readout.cursor.y})"]
    if readout.anchor is None or readout.dx is None or readout.dy is None:
        lines.append(f"{ANCHOR} —")
    else:
        lines.append(f"{ANCHOR} ({readout.anchor.x}, {readout.anchor.y})")
        lines.append(f"dx {readout.dx:+d}")
        lines.append(f"dy {readout.dy:+d}")
    for channel in readout.channels:
        if channel.relative is None:
            lines.append(f"{channel.name}  {channel.absolute}")
        else:
            lines.append(f"{channel.name}  {channel.absolute}  {channel.relative}")
    lines.append(readout.summary)
    return "\n".join(lines)


def status_text(state: ViewerState) -> str:
    image = state.image
    if image is None:
        return NO_IMAGE
    planes = ", ".join(
        f"{plane.name} {plane.bit_depth}bit {ORIGIN_LABEL[plane.origin]}" for plane in image.planes
    )
    cursor = f"({state.cursor.x}, {state.cursor.y})" if state.cursor is not None else "—"
    note = f"  {CONVERTED_NOTE}" if image.any_converted else ""
    needed = label_zoom(state)
    zoom_hint = f"  放到 {needed}× 显示数值" if needed is not None and state.zoom < needed else ""
    return (
        f"{image.path.name}  {image.width}×{image.height}  模式 {image.source_mode}  "
        f"帧 {image.frame_index + 1}/{image.frame_count}  {planes}  "
        f"{state.zoom:g}×  光标 {cursor}{note}{zoom_hint}"
    )
