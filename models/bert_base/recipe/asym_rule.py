"""통계 규칙으로 활성값 비대칭: 대칭 대비 scale 이 GAIN 배 이상 줄면 비대칭.
대칭 경계 M = 127 * scale (보정이 정한 값) 은 그대로, 짧은 쪽 끝만 보정 최솟/최댓값으로.
대상: 컴파일러가 지원하는 곳 = DQ 가 상수 가중치 MatMul 의 입력이고 LayerNorm 출력이 아닌 텐서.
--all 이면 지원 여부와 관계없이 규칙에 걸리는 모든 활성값 (가능성 확인용).
Usage: asym_rule.py <QDQ.onnx> <fp32 smoothed.onnx> <calib_ids.npy> <out.onnx> [--all]"""
import sys; sys.path.insert(0,"/tmp/pylibs")
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh, helper
src, fp32, calib, dst = sys.argv[1:5]; ALL = "--all" in sys.argv; GAIN = 1.25
m=onnx.load(src); g=m.graph; init={i.name:i for i in g.initializer}
cons={}; prod={}
for n in g.node:
    for i in n.input: cons.setdefault(i,[]).append(n)
    for o in n.output: prod[o]=n
qs=[n for n in g.node if n.op_type=="QuantizeLinear" and n.input[0] not in init]
def supported(q):
    if prod.get(q.input[0]) is not None and prod[q.input[0]].op_type=="LayerNormalization": return False
    for d in cons.get(q.output[0],[]):
        if d.op_type!="DequantizeLinear": return False
        for mm in cons.get(d.output[0],[]):
            if not (mm.op_type=="MatMul" and mm.input[0]==d.output[0] and mm.input[1] not in init and prod.get(mm.input[1]) is not None and prod[mm.input[1]].op_type=="DequantizeLinear" and prod[mm.input[1]].input[0] in init): return False
    return True
f=onnx.load(fp32); fnames={o for n in f.graph.node for o in n.output}
names=[q.input[0] for q in qs if q.input[0] in fnames]
for t in names: f.graph.output.append(helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
so=ort.SessionOptions(); so.log_severity_level=3; so.intra_op_num_threads=2; so.inter_op_num_threads=1
s=ort.InferenceSession(f.SerializeToString(),so,providers=["CPUExecutionProvider"]); del f
on=[o.name for o in s.get_outputs()]; lo={t:0.0 for t in names}; hi={t:0.0 for t in names}
for r in np.load(calib):
    n=int((r!=0).sum()); out=dict(zip(on,s.run(None,{s.get_inputs()[0].name:r[None]})))
    for t in names:
        v=out[t][0,:n] if out[t].ndim>=2 and out[t].shape[0]==1 and out[t].shape[1]>=n else out[t]
        lo[t]=min(lo[t],float(v.min())); hi[t]=max(hi[t],float(v.max()))
picked=0; skipped=[]
for q in qs:
    t=q.input[0]
    if t not in lo: continue
    sn,zn=q.input[1],q.input[2]; s_old=float(nh.to_array(init[sn])); M=127*s_old
    a,b=max(lo[t],-M),min(hi[t],M)          # 짧은 쪽 끝
    if -a <= b: L,H=a,M                       # 양수 쪽이 길다
    else: L,H=-M,b
    s_new=(H-L)/255.0; gain=s_old/s_new
    if gain < GAIN: continue
    sup=supported(q)
    if not sup and not ALL: skipped.append((t,gain)); continue
    zp=int(np.clip(round(-128-L/s_new),-128,127))
    for u in cons[sn]+cons[zn]: assert u is q or u.input[0]==q.output[0], (t,u.name)
    init[sn].CopyFrom(nh.from_array(np.array(s_new,np.float32),sn)); init[zn].CopyFrom(nh.from_array(np.array(zp,np.int8),zn))
    picked+=1; print(f"  비대칭 {'지원' if sup else '미지원'} {t[-60:]:60s} {gain:.2f}x zp {zp}")
for t,gn in skipped: print(f"  (규칙에 걸리지만 컴파일러 미지원이라 대칭 유지) {t[-60:]:60s} {gn:.2f}x")
onnx.save(m,dst); print(f"비대칭 {picked} 곳 -> {dst}")
