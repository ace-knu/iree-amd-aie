"""per-tensor 양자화 모델의 attn_out/FC1 가중치를 per-channel 모델의 것으로 바꾼다 (axis=1)."""
import sys, re; sys.path.insert(0,"/tmp/pylibs")
import onnx
from onnx import helper
src, pcm, dst = sys.argv[1:4]
m=onnx.load(src); pc=onnx.load(pcm); g=m.graph
pci={i.name:i for i in pc.graph.initializer}; init={i.name:i for i in g.initializer}; cons={}
for n in g.node:
    for i in n.input: cons.setdefault(i,[]).append(n)
k=0
for n in g.node:
    if n.op_type!="DequantizeLinear" or n.input[0] not in init or len(init[n.input[0]].dims)<2: continue
    mm=[c for c in cons[n.output[0]] if c.op_type=="MatMul"][0]
    if not re.search(r"(intermediate/dense)/MatMul", mm.output[0]): continue
    for name in n.input: init[name].CopyFrom(pci[name])
    del n.attribute[:]; n.attribute.append(helper.make_attribute("axis",1)); k+=1
onnx.save(m,dst); print(f"FC1 per-channel {k}개 추가 -> {dst}")
