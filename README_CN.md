## 预印本论文
描述当前模型的预印本论文可以在ChemRxiv上找到：[Prediction of Molecular Critical Properties Based on Boruta Feature Selection: Balancing Accuracy, Applicability and Explainability](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001485/v1)
BiBTeX引用条目：
```bibtex
@article{Li2026CriticalProperties,
      author = {Xurui Li  and Zhiguo Gan  and Jiaming Zhang  and Hongxi Zeng  and Zheng Liu  and Diannan Lu },
      title = {Prediction of Molecular Critical Properties Based on Boruta Feature Selection: Balancing Accuracy, Applicability and Explainability},
      journal = {ChemRxiv},
      volume = {2026},
      number = {0401},
      pages = {},
      year = {2026},
      doi = {10.26434/chemrxiv.15001485/v1},
      URL = {https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15001485/v1},
      eprint = {https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15001485/v1}
}
```

## 代码库介绍
*   本代码库为`Prediction of Molecular Critical Properties Based on Boruta Feature Selection: Balancing Accuracy, Applicability and Explainability`论文的代码库，包含了测试集数据、模型和测试结果等相关内容。
*   由于ThermoData Engine(TDE)数据库为商业数据库，本研究仅用于学术目的，未经TDE允许不可禁止公开测试数据集的临界数据点值，因此测试数据集仅包含分子名称和SMILES字符串，临界数据点实验数据已经被手动隐去。由于我们公开的为原始计算代码，因此计算时可能会因为没有数据而出现AARD为nan的结果，但这并不影响代码的正常运行。

## 文件夹介绍
*   `data`文件夹包含特征归一化参数。
*   `homo`文件夹包含同系物临界点预测的结果。
      - `PURE-homo-150.xlsx`包含PONA四类烃类同系物，各150个点。
*   `image`文件夹包含模型训练验证集效果图。
*   `model`文件夹包含训练好的模型参数。
*   `newexp`文件夹包含模型在新物质测试的预测效果。
      - `TEST_A.xlsx`包含Test(A)的测试数据。
      - `TEST_AB.xlsx`包含Test(A)和Test(B)的测试数据。

## ipynb文件介绍
*   `GNN_predict.ipynb`使用GNN模型在Test(A)进行预测。
*   `GRP_Pc_2Dpredict.ipynb`使用GRP(2D)模型在Test(A)上进行$P_c$预测，内部输出分别为Test(A)、Test(ALL)和Test(B)上的预测指标。
*   `GRP_Tc_2Dpredict.ipynb`使用GRP(2D)模型在Test(A)上进行$T_c$预测，内部输出分别为Test(A)、Test(ALL)和Test(B)上的预测指标。
*   `RF_Pc_2Dpredict.ipynb`使用RF模型在Test(A)上进行$P_c$预测，内部输出分别为Test(A)、Test(ALL)和Test(B)上的预测指标。
*   `RF_Tc_2Dpredict.ipynb`使用RF模型在Test(A)上进行$T_c$预测，内部输出分别为Test(A)、Test(ALL)和Test(B)上的预测指标。
*   `SVR_Pc_2Dpredict.ipynb`使用SVR模型在Test(A)上进行$P_c$预测，内部输出分别为Test(A)、Test(ALL)和Test(B)上的预测指标。
*   `SVR_Tc_2Dpredict.ipynb`使用SVR模型在Test(A)上进行$T_c$预测，内部输出分别为Test(A)、Test(ALL)和Test(B)上的预测指标。
*   `predict_3D.ipynb`为全部3D模型在测试集上的预测代码，内部分别输出GPR、RF和SVR模型在TEST(ALL)和TEST(A)上的预测指标。

## 测试数据介绍
*   测试集均来源于ThermoData Engine(TDE)数据库，包含Test(A)和Test(B)两部分。
*   Test(A)包含148个分子，Test(B)包含38个分子。
*   全部测试集均为一个一个人工查询得到的，因为TDE软件不允许批量查询或批量导出数据。
*   测试集中全部数据使用TDE评估接受的临界数据点，若有多个实验数据点，则取其平均值。
*   由于TDE为商业数据库，本研究仅用于学术目的，未经TDE允许不可禁止公开测试数据集的临界数据点值。
