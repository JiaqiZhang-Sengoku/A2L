# A²L：三域固定入口（Domain1/2无对比）

保留 UPA → PAUH → LGMC 三个核心模块、统一公开源模型和原论文两幅图的逻辑。本次停止自动实验，只保存各域固定 seed42、已完成30轮的最高验证分实现。

## 固定入口与最佳结果

| 入口 | 固定实现 | best轮 | Mean Dice % |
|---|---|---:|---:|
| A2L_Domain1.py | 正类原型回归；仅校准伪标签分割为BCE，真实标签及warmup仍为BCE+Dice；固定LR | 11 | 91.263741 |
| A2L_Domain2.py | 正类原型回归；BCE+Dice；校准后期余弦LR | 13 | 87.187745 |
| A2L_Domain4.py | 原CLASSIFICATION联合适配与C/O测试流程，未改方法 | 不按验证选轮，使用末轮 | 本次未运行 |

| 域 | OD Dice % ↑ | OC Dice % ↑ | OD ASSD px ↓ | OC ASSD px ↓ |
|---|---:|---:|---:|---:|
| Domain1 | 96.853772 | 85.673710 | 3.389613 | 9.207024 |
| Domain2 | 93.651467 | 80.724023 | 5.443396 | 7.587144 |

两域均无负类对比或新增一致性损失。golden原型在本步优化后用缓存特征更新，teacher参数固定EMA=0.98。D2第1–11轮lr保持0.0005，第12–30轮为 `lr0*(1+cos(pi*(e-11)/20))/2`，e从1开始；D1固定lr。没有trick搜索开关。

“最好”指固定seed42的候选最高分，不是逐指标或跨seed拼接。D1新版本只有一次完整seed42；D2余弦三seed均值86.0586%，低于固定LR的86.1758%。这里保存已观察最高分代码，不宣称稳定涨点。全部为独立TRAIN验证，正式test尚未生成。

开发服务器上的最佳权重入口为 `Outputs/Best/Domain1/best_student.pth.tar` 和 `Outputs/Best/Domain2/best_student.pth.tar`。GitHub仓库不分发数据集、源模型、实验权重或输出；这些内容受体积与各自许可约束，需要在本地单独准备。

## 目录职责

```text
A2L.py                 共享训练流程和三个核心模块
A2L_Domain1.py          Domain1固定入口
A2L_Domain2.py          Domain2固定入口
A2L_Domain4.py          Domain4联合适配与开放域测试固定入口
A2L_Test.sh             串行启动三域完整流程（会训练，不是只读测试）
Config/                三域配置和固定验证名单
Networks/              MobileNetV2 + DeepLabv3+
Dataloaders/            标签读取、weak/strong外观增强
Utils/                 原型回归、指标、路径和运行记录
Data/                  本地实验数据（不纳入Git）
Checkpoints/           本地源模型与预训练资料（不纳入Git）
Outputs/               本地训练结果（不纳入Git）
```

PLS/TAR独立脚本、废弃策略及编辑器备份不进入默认方法。仓库中的 `A2L_Test.sh` 会串行启动三域完整流程，不是只读检查命令；仅在明确需要重新训练全部实验时运行。

## 数据与源模型

将数据组织为 `Data/DomainX/{train,test}/ROIs/{image,mask}`。Domain1和Domain2需要train/test，Domain4只作为开放域测试集。将统一源模型放在 `Checkpoints/Source/source_model.pth.tar`。上述目录均由 `.gitignore` 排除，不会被提交。

## 使用

在项目根目录查看参数：

```bash
python -B A2L_Domain1.py --print-config
python -B A2L_Domain2.py --print-config
python -B A2L_Domain4.py --print-config
```

只有以后明确需要重训时才使用 `python -B A2L_Domain1.py` 或 `python -B A2L_Domain2.py`。默认30轮、warmup10、batch8、lr0.0005、seed42；使用验证并跳过test。best仅从11–30轮按student Mean Dice最高选择，同分取有效ASSD低、再早轮。新输出目录不覆盖历史。不支持只加载student权重的精确续训。

D1适配40/验证10/查询2，D2适配79/验证20/查询5。验证标签不进入梯度，未查询mask不解码。额外验证成本10/20张；累计开发唯一标注16/36张。temperature及cal_margin_scale仅对Domain4分类式生效；回归保留相关统计遍历以维持已验证随机流。

原始精确对照使用strict deterministic backend、CUBLAS_WORKSPACE_CONFIG=:4096:8及记录的TF32设置；普通入口不承诺逐位复现。对应内部核验记录和大体积实验产物未随代码仓库发布。本次只验证CPU关键路径和接口等价，不重新训练30轮。

## Domain4不变

新增固定入口 `python -B A2L_Domain4.py`，等价调用原 `python -B A2L.py --config Config/A2L_Domain4.json` 流程。默认30轮、warmup10、全局查询7张，保留原CLASSIFICATION公式：D2→D1组成149张池，一个共享模型，测试C域D2/D1与O域D4。D4不进训练，本次不运行。原D4公式仍含原型分类损失，不属于D1/D2无对比替代范围。

Domain4入口自动读取Config/A2L_Domain4.json，拒绝配置或CLI切换为其他域，不自动添加单域验证划分或跳过test；显式使用--skip-test可关闭测试，--validation-split仍不支持。默认在Outputs/Domain4下创建新时间戳目录，保存末轮student/teacher与metrics.csv、compound_eval_summary.csv，没有基于test挑选最佳轮。使用--help或--print-config不会训练、读取数据或加载权重。

建议用对应单域wrapper；A2L.py按dataset分派D1/D2无对比与D4原分类式流程。单域wrapper只做验证；A2L.py的test流程评估末轮，不能当作best checkpoint的test结果。不要用--model-file加载best来当评估命令，该参数仍会进入训练。

输入512×512，OC=视杯、OD=视盘；保留原0.75评估阈值、Dice平滑及空mask排除ASSD规则。依赖Python3.12、PyTorch2.8.0+CUDA12.8、torchvision、NumPy、SciPy、Pillow、OpenCV、MedPy。
