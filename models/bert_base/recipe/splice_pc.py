"""per-tensor QDQ 모델의 가중치를 per-channel 모델의 것으로 바꾼다 (axis=1).
--skip-head-split: 출력이 Reshape->Transpose 로 헤드를 나누는 matmul(Q/K/V)은 건너뛴다.
배포 경로의 head split 이 per-channel scale 을 실을 수 없어서다 (구조로 판단, 이름 안 봄)."""
import sys; sys.path.insert(0,"/tmp/pylibs")
import onnx
from onnx import helper
src, pcm, dst = sys.argv[1:4]; skip = "--skip-head-split" in sys.argv
m=onnx.load(src); pc=onnx.load(pcm); g=m.graph
pci={i.name:i for i in pc.graph.initializer}; init={i.name:i for i in g.initializer}; cons={}
for n in g.node:
    for i in n.input: cons.setdefault(i,[]).append(n)
def head_split(mm):
    q=cons.get(mm.output[0],[])
    q=[c for c in q if c.op_type=="QuantizeLinear"]
    t=q[0].output[0] if q else mm.output[0]
    d=[c for c in cons.get(t,[]) if c.op_type=="DequantizeLinear"]
    t=d[0].output[0] if d else t
    # Q/K/V: bias Add -> Reshape -> Transpose
    for a in cons.get(t,[]):
        for r in cons.get(a.output[0],[]) if a.op_type=="Add" else [a]:
            if r.op_type=="Reshape" and any(x.op_type=="Transpose" for x in cons.get(r.output[0],[])): return True
    return False
k=s=0
for n in g.node:
    if n.op_type!="DequantizeLinear" or n.input[0] not in init or len(init[n.input[0]].dims)<2: continue
    mm=[c for c in cons[n.output[0]] if c.op_type=="MatMul"][0]
    if skip and head_split(mm): s+=1; continue
    for name in n.input: init[name].CopyFrom(pci[name])
    del n.attribute[:]; n.attribute.append(helper.make_attribute("axis",1)); k+=1
onnx.save(m,dst); print(f"per-channel {k}개, head split 이라 per-tensor 유지 {s}개 -> {dst}")
