"""QDQ 모델(접기 전)의 GELU 출력(SmoothQuant 의 sq_x_N, FC2 입력) 양자화를 비대칭 int8 로 바꾼다.
위쪽 끝은 지금 값(127 * scale, 백분위 보정)을 그대로 두고, 아래쪽 끝을 fp32 SmoothQuant 모델의
보정 문장 최솟값으로 잡아 [lo, hi] 를 256 칸에 나눈다.
Usage: asym_q.py <in QDQ.onnx> <fp32 smoothed s.onnx> <calib_ids.npy> <out.onnx>"""
import sys, re; sys.path.insert(0,"/tmp/pylibs")
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh, helper
src, fp32, calib, dst = sys.argv[1:5]
m=onnx.load(src); g=m.graph; init={i.name:i for i in g.initializer}
qs=[n for n in g.node if n.op_type=="QuantizeLinear" and re.fullmatch(r"sq_x_\d+", n.input[0])]
names=[q.input[0] for q in qs]
users={}
for n in g.node:
    for i in n.input: users.setdefault(i,[]).append(n)
# fp32 모델에서 보정 최솟값
f=onnx.load(fp32)
for t in names: f.graph.output.append(helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
so=ort.SessionOptions(); so.log_severity_level=3; so.intra_op_num_threads=2; so.inter_op_num_threads=1
s=ort.InferenceSession(f.SerializeToString(),so,providers=["CPUExecutionProvider"]); del f
on=[o.name for o in s.get_outputs()]; lo={t:0.0 for t in names}
for r in np.load(calib):
    n=int((r!=0).sum()); out=dict(zip(on,s.run(None,{s.get_inputs()[0].name:r[None]})))
    for t in names: lo[t]=min(lo[t], float(out[t][0,:n].min()))
for q in qs:
    t=q.input[0]; sn, zn = q.input[1], q.input[2]
    # 스케일·영점 초기값은 이 텐서의 Q/DQ 만 써야 한다
    for u in users[sn]+users[zn]:
        assert u.op_type in ("QuantizeLinear","DequantizeLinear") and (u is q or u.input[0]==q.output[0]), (t, u.name)
    s_old=float(nh.to_array(init[sn])); assert int(nh.to_array(init[zn]))==0
    hi=127*s_old; s_new=(hi-lo[t])/255.0; zp=int(np.clip(round(-128-lo[t]/s_new),-128,127))
    init[sn].CopyFrom(nh.from_array(np.array(s_new,np.float32), sn))
    init[zn].CopyFrom(nh.from_array(np.array(zp,np.int8), zn))
    print(f"{t:8s} lo {lo[t]:7.3f} hi {hi:7.3f}  scale {s_old:.5f} -> {s_new:.5f} ({s_old/s_new:.2f}x)  zp {zp}")
onnx.save(m,dst); print(f"{len(qs)} 곳 비대칭 -> {dst}")
