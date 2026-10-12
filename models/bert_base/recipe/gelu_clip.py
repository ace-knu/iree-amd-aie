"""소비자 포화 규칙: 양자화된 활성값을 쓰는 곳이 GELU 하나뿐이면, 하한을 GELU 포화점 -C 로 올린다.
상한(127·scale)은 그대로 두고 비대칭으로 다시 잡는다. GELU 는 ONNX 에서
x·0.5·(1+erf(x/√2)) 로 풀려 있으므로, DQ 출력의 소비자가 모두 이 모양 안에 있는지 본다.
python3 gelu_clip.py in.onnx out.onnx [C=4]"""
import sys; sys.path.insert(0,"/tmp/pylibs")
import numpy as np, onnx
from onnx import numpy_helper as nh
src,dst=sys.argv[1],sys.argv[2]; C=float(sys.argv[3]) if len(sys.argv)>3 else 4.0
m=onnx.load(src); g=m.graph; init={i.name:i for i in g.initializer}
cons={}
for n in g.node:
    for i in n.input: cons.setdefault(i,[]).append(n)
def only(v):
    c=cons.get(v,[]); return c[0] if len(c)==1 else None
def is_gelu(x):
    """x 의 소비자가 Div(x,√2) -> Erf -> Add(1) -> Mul(x,·) 과 그 Mul 뿐인가."""
    c=cons.get(x,[])
    if len(c)!=2 or {n.op_type for n in c}!={"Div","Mul"}: return False
    div=next(n for n in c if n.op_type=="Div"); mul=next(n for n in c if n.op_type=="Mul")
    erf=only(div.output[0]); add=erf and erf.op_type=="Erf" and only(erf.output[0])
    return bool(add and add.op_type=="Add" and add.output[0] in mul.input)
k=0
for q in [n for n in g.node if n.op_type=="QuantizeLinear" and n.input[0] not in init]:
    dqs=cons.get(q.output[0],[])
    if not dqs or any(d.op_type!="DequantizeLinear" or not is_gelu(d.output[0]) for d in dqs): continue
    if any(q.input[0]==o for o in g.output): continue
    s0=float(nh.to_array(init[q.input[1]])); z0=int(nh.to_array(init[q.input[2]]))
    if z0!=0 or 128*s0<=C: continue              # 이미 비대칭이거나 포화점까지 안 가는 경우
    hi=127*s0; lo=-C; s=(hi-lo)/255; z=int(np.clip(round(-128-lo/s),-128,127))
    for d in dqs:                                  # Q 와 DQ 가 scale/zp 를 같이 쓰는지 확인
        assert d.input[1]==q.input[1] and d.input[2]==q.input[2]
    zt=nh.to_array(init[q.input[2]]).dtype
    init[q.input[1]].CopyFrom(nh.from_array(np.array(s,np.float32),q.input[1]))
    init[q.input[2]].CopyFrom(nh.from_array(np.array(z,zt),q.input[2]))
    k+=1; print(f"{q.input[0]}: scale {s0:.4f} -> {s:.4f} ({s/s0:.2f}배), zp {z}")
onnx.save(m,dst); print(f"{k}곳 -> {dst}")
