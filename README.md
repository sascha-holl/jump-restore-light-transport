# **Jump Restore Light Transport**

## **Overview**
This repository contains implementations of the **Jump Restore Light Transport** algorithm, proposed in the paper [**Jump Restore Light Transport**](https://restore-light-transport.mpi-inf.mpg.de/), integrated into two different existing rendering systems:
- [**pbrt-v4**](https://github.com/mmp/pbrt-v4) (licensed under Apache License 2.0)
- [**Langevin-MCMC**](https://github.com/luanfujun/Langevin-MCMC) (licensed under MIT License), which is based on [**dpt**](https://github.com/BachiLi/dpt)

The implementations can be found in the following folders:
- **pbrt-v4/**: Integration of our algorithm into the pbrt-v4 rendering system.
- **dpt/**: Integration of our algorithm into the dpt rendering system.

---

## **License**

This project is **dual-licensed**, with different licenses applied to different parts of the project.

### **1. Original Projects**
- **pbrt-v4**:
  - The original [pbrt-v4](https://github.com/mmp/pbrt-v4) code is licensed under the [Apache License 2.0](LICENSE.Apache-2.0).
  - Please refer to the `pbrt-v4/` folder for our modifications.

- **Langevin-MCMC**:
  - The original [Langevin-MCMC](https://github.com/luanfujun/Langevin-MCMC) code is licensed under the [MIT License](LICENSE.MIT).
  - Please refer to the `dpt/` folder for our modifications.

### **2. Our Algorithm and Modifications**
All modifications made by Sascha Holl are dual-licensed:

- **Open Source License**:
  - You may use, modify, and distribute the algorithm under the terms of the [Apache License 2.0](LICENSE).
  - This applies to all changes and additions made by Sascha Holl in both the `pbrt-v4/` and `dpt/` folders.

- **Commercial License**:
  - If you wish to use the algorithm for commercial purposes, you must obtain a commercial license. Please see [COMMERCIAL_LICENSE](COMMERCIAL_LICENSE) for details.
  - For commercial inquiries, contact Sascha Holl at sascha.holl@gmail.com.

## **SPDX License Identifiers**
The following SPDX license identifiers are used throughout the codebase:
- `SPDX-License-Identifier: Apache-2.0` (for open-source code derived from pbrt-v4)
- `SPDX-License-Identifier: MIT` (for open-source code derived from Langevin-MCMC)
- `SPDX-License-Identifier: Apache-2.0 OR Commercial` (for our dual-licensed modifications)

## How to Use This Repository

To use the algorithm in either system, navigate to the respective folders (pbrt-v4/ or dpt/) and follow the installation instructions provided in the corresponding README.md files.

For more details on the algorithm, see the [project page](https://restore-light-transport.mpi-inf.mpg.de/) of the corresponding research paper.

## Contact

For questions, issues, or commercial licensing, please reach out to sascha.holl@gmail.com.

## **How to Cite**
If you use these implementations in your research, please cite this work:

```plaintext
@software{holl2024jrltcode,
  author = {Sascha Holl},
  title  = {Jump Restore Light Transport - pbrt-v4 and dpt implementation},
  year   = {2024},
  url    = {https://github.com/yourusername/yourproject}
}
```
## Acknowledgments

Special thanks to the original authors of both pbrt-v4 and dpt for their outstanding contributions to the open-source community.
