"""A19 交付件：裁切到人物、加字幕、出分镜图。

调用顺序
    final.mp4（1080²渲染） → final_cropped.mp4（裁到人物、缩到720）
                          → final_captioned.mp4（加段号与进度）
                          → 分镜.png（每段一帧）
"""
from pathlib import Path


def deliver(job, job_id):
    from hm import frame_crop
    from hm.job_report import report
    from hm.annotate_video import annotate

    # job_report 与 annotate_video 在工程根目录，按 job_id 找 outputs 目录
    crops = frame_crop.crop_square(job_id)
    annotate(job_id)
    try:
        report(job_id)
    except Exception as e:
        print("  分镜图生成失败：%s" % str(e)[:100])
    return crops
