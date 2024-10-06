  
We provide compressed packages of all available data for [download](https://cloud.tsinghua.edu.cn/d/5e641bc7b6ee46b785ee/). Feel free to download based on your needs, and then put them here.
  
  - [process_dataset] contains the pre-processed CrossDock dataset. Following the similar process by [Luoshi Tong](https://github.com/luost26/3D-Generative-SBDD/tree/main/data), we retrieve 4 reference molecules for each sample.
  - [retrieval_database] contains the several databases for retrieving. In each database directory, `total.smi` provides the complete SMILES sequences, and `mol` contains the pre-processed features used for docking in [FABind](https://github.com/QizhiPei/FABind). Data is stored in key-value pairs in `database.pth`, where each key corresponds to a file in `mol`, and the value contains the drug features predicted by [ConPLex](https://github.com/samsledje/ConPLex). Note that due to the large size of the `GEOM` and `ZINC` databases, we randomly sample 100,000 molecules to construct their `mol` and `database.pth`.
  - [test_set] contains the test set for comparison in paper.
