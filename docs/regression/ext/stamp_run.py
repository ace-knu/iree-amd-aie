#!/usr/bin/env python3
"""명령을 실행하며 stderr 줄마다 받은 시각(ms, 시작 기준)을 붙인다(V5 순서 검사용).
10/18 docs/2026-10-18/2_…/02_…/run_timed.py와 같은 방식이고, fd 표본은 뺐다.
  stamp_run.py <stamped_stderr> <stdout> -- <command...>   종료 코드는 명령의 것을 그대로 돌려준다."""
import sys, time, subprocess
st, so = sys.argv[1], sys.argv[2]; cmd = sys.argv[sys.argv.index('--') + 1:]
t0 = time.monotonic()
p = subprocess.Popen(cmd, stdout=open(so, 'w'), stderr=subprocess.PIPE)
with open(st, 'w') as f:
    for raw in p.stderr: f.write(f'{(time.monotonic() - t0) * 1e3:.3f}\t{raw.decode(errors="replace").rstrip()}\n')
sys.exit(p.wait())
