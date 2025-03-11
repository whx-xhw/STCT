# Self-correcting Clustering
This is the source code of Self-correcting Clustering.

## Abstract

Learning from noisy labels (LNL) aims to train high-performance deep models using noisy datasets. Meta learning based label correction methods have demonstrated remarkable performance in LNL by designing various meta label rectification tasks. However, extra clean validation set is a prerequisite for these methods to perform label correction, requiring extra labor and greatly limiting their practicality. To tackle this issue, we propose a novel noisy meta label correction framework STCT, which counterintuitively uses noisy data to correct label noise, borrowing the spirit in the saying ``\textbf{ \underline{S}}et a \textbf{ \underline{T}}hief to \textbf{ \underline{C}}atch a \textbf{ \underline{T}}hief''. The core idea of STCT is to leverage noisy data which is i.i.d. with the training data as a validation set to evaluate model performance and perform label correction in a meta learning framework, eliminating the need for extra clean data. By decoupling the complex bi-level optimization in meta learning into representation learning and label correction, STCT is solved through an alternating training strategy between noisy meta correction and semi-supervised representation learning. Extensive experiments on synthetic and real-world datasets demonstrate the outstanding performance of STCT, particularly in high noise rate scenarios. STCT achieves 96.9\% label correction and 95.2\% classification performance on CIFAR-10 with 80\% symmetric noise, significantly surpassing the current state-of-the-art.
![Main Image](/img/fig.PNG)
