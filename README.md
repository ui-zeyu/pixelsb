# pixelsb

像素级图像查看器，用来看通道数值、相对锚点的偏移，以及单个通道的某一位。

```bash
uv run pixelsb
uv run pixelsb image.png
```

单击图像设置锚点。右侧格子里，单击勾选或取消任意通道的任意位，选中的位保留在原来的通道和权重上；⌘、Ctrl 或 Shift 加单击＝只看这一位。放大到格子放得下文字时，数字直接画在像素上，字号随格子自适应。`+` / `-` 缩放，`看清数值` 放到刚好能看见数字。菜单「帮助 → 快捷键」里有完整列表。

调色板图像会同时给出原始索引和查色后的 R/G/B/A。16-bit 灰度按 0–15 位显示。CMYK 等模式会先转成 RGBA，状态栏会标明位平面来自转换后的数据。

```bash
uv run ruff format
uv run ruff check
uv run ty check
uv run pytest
```
