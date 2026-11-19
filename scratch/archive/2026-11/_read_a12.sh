#!/bin/bash
cd /home/chuan/motion-work
awk '/def step_a12_a13/,/job.save_json\("storyboard/' hm/run_job.py | head -n 70
