"""SmoothQuant α 를 보정 데이터만으로 고른다. 위치마다 α 후보별로, 보정 문장에서
Y = X W (fp32) 와 레시피대로 양자화한 Q(X/s) Q(diag(s) W) 의 상대 오차를 재고 가장 작은 α.
레시피: 가중치 per-channel 대칭 int8, 활성값 per-tensor 백분위 99.999 대칭 int8,
통계 규칙(scale 1.25 배 이상 감소, LayerNorm 출력 제외)이면 비대칭.
Usage: alpha_rule.py <fp32.onnx> <calib_ids.npy> <out.json>"""
import sys, json; sys.path.insert(0,"/tmp/pylibs")
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh, helper
src, calib, dst = sys.argv[1:4]
GRID=[0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]; PCT=99.999; GAIN=1.25
ELEMENTWISE={"Mul","Add","Sub","Div","Erf","Relu","Sigmoid","Tanh","Gelu","Neg","Exp","Pow","Sqrt","Clip","HardSigmoid","HardSwish","LeakyRelu","Elu","Softplus"}
m=onnx.load(src); g=m.graph; init={i.name:i for i in g.initializer}; cons={}
for n in g.node:
    for i in n.input: cons.setdefault(i,[]).append(n)
sites=[]
for n in g.node:
    t=n.output[0]; mms=[c for c in cons.get(t,[]) if c.op_type=="MatMul" and c.input[0]==t and c.input[1] in init]
    if mms and (n.op_type=="LayerNormalization" or n.op_type in ELEMENTWISE):
        sites.append((t, n.op_type=="LayerNormalization", mms))
tap=onnx.ModelProto(); tap.CopyFrom(m)
for t,_,_ in sites: tap.graph.output.append(helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
so=ort.SessionOptions(); so.log_severity_level=3; so.intra_op_num_threads=2; so.inter_op_num_threads=1
s=ort.InferenceSession(tap.SerializeToString(),so,providers=["CPUExecutionProvider"]); del tap
names=[t for t,_,_ in sites]; rows={t:[] for t in names}; amax={t:0 for t in names}
for r in np.load(calib):
    n=int((r!=0).sum())
    for t,v in zip(names, s.run(names,{"input_ids":r[None]})):
        v=v.reshape(-1,v.shape[-1]); amax[t]=np.maximum(amax[t],np.abs(v).max(0)); rows[t].append(v[:n])
def qerr(X,W,is_ln,a,am):
    wmax=np.abs(W).max(1)
    sc=(np.power(np.maximum(am,1e-5),a)/np.power(np.maximum(wmax,1e-5),1-a)).astype(np.float32)
    Xs=X/sc; Ws=W*sc[:,None]
    sw=np.abs(Ws).max(0)/127; Wq=np.clip(np.round(Ws/sw),-127,127)*sw
    M=np.percentile(np.abs(Xs),PCT); sx=M/127
    lo,hi=max(float(Xs.min()),-M),min(float(Xs.max()),M)
    L,H=(lo,M) if -lo<=hi else (-M,hi); s2=(H-L)/255
    if not is_ln and sx/s2>=GAIN:
        zp=np.clip(np.round(-128-L/s2),-128,127); Xq=(np.clip(np.round(Xs/s2)+zp,-128,127)-zp)*s2
    else: Xq=np.clip(np.round(Xs/sx),-128,127)*sx
    Y=X@W; return float(((Xq@Wq-Y)**2).sum()/(Y**2).sum())
table={}
for t,is_ln,mms in sites:
    X=np.concatenate(rows[t]).astype(np.float32); W=np.concatenate([nh.to_array(init[mm.input[1]]) for mm in mms],1).astype(np.float32)
    table[t]={a:qerr(X,W,is_ln,a,amax[t]) for a in GRID}; rows[t]=None
    best=min(GRID,key=lambda a:table[t][a])
    print(f"{('LN' if is_ln else 'act')} {t[-50:]:50s} best α {best}  " + " ".join(f"{a}:{table[t][a]:.2e}" for a in GRID), flush=True)
per={t:min(GRID,key=lambda a:table[t][a]) for t in names}
tot={a:sum(table[t][a] for t in names) for a in GRID}; glob=min(GRID,key=lambda a:tot[a])
print("전체 합 기준 α:", glob, " ".join(f"{a}:{tot[a]:.3e}" for a in GRID))
json.dump({"per_site":per,"global":glob,"table":{t:{str(a):v for a,v in d.items()} for t,d in table.items()}},open(dst,"w"),indent=1)
