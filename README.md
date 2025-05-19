# MonoECls: A Monolithic Lightweight LLM is Only All You Need for Chinese Literary Classification

## InternLM2_embedding / bert_embedding / roberta_embedding
The `rep_extract.py` uses language model to extract the representation of `dataset` and saves the representation as `.pt` file.

## Setup Environment

Before running this project, you need to create a conda environment and install required packages. <br>

```bash 
conda create -n MonoECls python=3.11.8
conda activate MonoECls
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Data and trained models

| Name | Link             |
|------|------------------|
| AuthIDE | [huggingface]() |
|TextCls|   [huggingface]() |
|TTCls |   [huggingface]() |
|PoemSen |   [huggingface]() |

## Acknowledgements

This project refers to the following open-source projects, and we would like to express our gratitude to the related projects and developers.

- 书生（InternLM）：[https://github.com/QwenLM/Qwen](https://github.com/InternLM/InternLM)

```
@misc{cai2024internlm2,
      title={InternLM2 Technical Report},
      author={Zheng Cai and Maosong Cao and Haojiong Chen and Kai Chen and Keyu Chen and Xin Chen and Xun Chen and Zehui Chen and Zhi Chen and Pei Chu and Xiaoyi Dong and Haodong Duan and Qi Fan and Zhaoye Fei and Yang Gao and Jiaye Ge and Chenya Gu and Yuzhe Gu and Tao Gui and Aijia Guo and Qipeng Guo and Conghui He and Yingfan Hu and Ting Huang and Tao Jiang and Penglong Jiao and Zhenjiang Jin and Zhikai Lei and Jiaxing Li and Jingwen Li and Linyang Li and Shuaibin Li and Wei Li and Yining Li and Hongwei Liu and Jiangning Liu and Jiawei Hong and Kaiwen Liu and Kuikun Liu and Xiaoran Liu and Chengqi Lv and Haijun Lv and Kai Lv and Li Ma and Runyuan Ma and Zerun Ma and Wenchang Ning and Linke Ouyang and Jiantao Qiu and Yuan Qu and Fukai Shang and Yunfan Shao and Demin Song and Zifan Song and Zhihao Sui and Peng Sun and Yu Sun and Huanze Tang and Bin Wang and Guoteng Wang and Jiaqi Wang and Jiayu Wang and Rui Wang and Yudong Wang and Ziyi Wang and Xingjian Wei and Qizhen Weng and Fan Wu and Yingtong Xiong and Chao Xu and Ruiliang Xu and Hang Yan and Yirong Yan and Xiaogui Yang and Haochen Ye and Huaiyuan Ying and Jia Yu and Jing Yu and Yuhang Zang and Chuyu Zhang and Li Zhang and Pan Zhang and Peng Zhang and Ruijie Zhang and Shuo Zhang and Songyang Zhang and Wenjian Zhang and Wenwei Zhang and Xingcheng Zhang and Xinyue Zhang and Hui Zhao and Qian Zhao and Xiaomeng Zhao and Fengzhe Zhou and Zaida Zhou and Jingming Zhuo and Yicheng Zou and Xipeng Qiu and Yu Qiao and Dahua Lin},
      year={2024},
      eprint={2403.17297},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

- LLaMA-Factory：https://github.com/hiyouga/LLaMA-Factory

```
@article{zheng2024llamafactory,
  title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models}, 
  author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Yongqiang Ma},
  journal={arXiv preprint arXiv:2403.13372},
  year={2024},
  url={http://arxiv.org/abs/2403.13372}
}
```
---
