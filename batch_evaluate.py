import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import sys
import logging

from v2vbench import EvaluatorWrapper

logging.basicConfig(
    level=logging.INFO,  # Change it to logging.DEBUG if you want to troubleshoot
    format="%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
    handlers=[logging.StreamHandler(sys.stdout),]
)

evaluation = EvaluatorWrapper(metrics='all')  # 'all' for all metrics
# print(EvaluatorWrapper.all_metrics)  # list all available metrics

results = evaluation.evaluate(
    edit_video='/home/chx/mySrc/HunyuanVideo_rf_inv/results/V2VBench/rf-inversion',#编辑视频路径, 在batch edit是设置
    reference_video='/data/chx/V2VBench/videos',#参考视频路径，在V2VBench数据集中有提供
    index_file='/data/chx/V2VBench/config.yaml',#配置文件路径, 在V2VBench数据集中有提供
    output_dir='./results',
    # it is recommended to cache the flow of source videos for motion_alignment to avoid redundant computation
    evaluator_kwargs={'motion_alignment': {'cache_flow': True}},
)