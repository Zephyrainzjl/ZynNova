# ZynNova ZynMorph 电极微结构测试补丁

本补丁包含：

- `notebooks/ZynNova_Electrode_Microstructure_Test.ipynb`：完整电极微结构测试；
- `src/zynnova/zynmorph/reconstruction.py`：修正二维切片沿法向广播时错误放大平面尺寸的问题；
- `tests/zynnova/test_zynmorph_reconstruction.py`：覆盖 axis=0/1/2、低分辨率切片、非法 prior 和未知相标签的回归测试。

将补丁目录中的文件按相同相对路径覆盖到 ZynNova 仓库即可。然后运行：

```bash
cd /path/to/ZynNova
python -m pip install -e ".[zynnova]"
python -m pip install pandas matplotlib
PYTHONPATH=src pytest -q tests/zynnova
jupyter lab notebooks/ZynNova_Electrode_Microstructure_Test.ipynb
```

验证结果：

```text
17 passed
```
