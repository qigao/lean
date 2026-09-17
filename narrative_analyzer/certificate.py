"""Deterministic closed-template Lean certificate generation."""
from __future__ import annotations
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
import hashlib, json
from types import MappingProxyType
from .model import ConstantSchedule, ExactRat, NamedSchedule, PathModel, PiecewiseSchedule
from .named_schedules import LEAN_NAMESPACE, FixedFixtureRoute, fixed_fixture_route, resolve_named_schedule
from .result import CertificateGenerationError, ClaimStatus
from ._certificate_path2 import named_path2_body

PROD="NarrativeDynamics.Core.FitnessABMPathNExposureConvergence"
EXP="NarrativeDynamics.FitnessABMPathNExposure"
CONV="NarrativeDynamics.FitnessABMPathNExposureConvergence"
STRUCT=("parameters_valid","initial_all_broadcast","exposure_law","effective_alpha_lookup")
NEG=("parameters_valid","initial_all_broadcast")
VALID={s:f"{LEAN_NAMESPACE}.{s}_valid" for s in ("slowZeroSchedule","nearOneSchedule","harmonicSchedule")}

@dataclass(frozen=True)
class CertificateClaim:
    claim_id:str; expected_status:ClaimStatus; theorem:str
    assumptions:tuple[str,...]; exact_values:Mapping[str,str]
    def __post_init__(self):
        if not self.claim_id or not self.theorem.strip() or self.expected_status is ClaimStatus.UNKNOWN: raise ValueError("invalid certificate claim")
        object.__setattr__(self,"assumptions",tuple(self.assumptions)); object.__setattr__(self,"exact_values",MappingProxyType(dict(self.exact_values)))
@dataclass(frozen=True)
class Certificate:
    source:str; claims:tuple[CertificateClaim,...]
    def __post_init__(self): object.__setattr__(self,"claims",tuple(self.claims))

def F(q): return Fraction(q.numerator,q.denominator)
def rat(q): return f"({q.numerator} : Rat)" if q.denominator==1 else f"(({q.numerator} : Rat) / {q.denominator})"
def txt(q): return str(q.numerator) if q.denominator==1 else f"{q.numerator}/{q.denominator}"
def schedule_values(m):
    s=m.schedule
    if isinstance(s,ConstantSchedule): return (s.value,)
    if isinstance(s,PiecewiseSchedule): return (s.default,*(v for _,v in s.points))
    return ()
def candidate_global_interior(m):
    vals=schedule_values(m)
    if not vals:return None
    x=min([z for q in vals for z in (F(q),1-F(q))])
    return None if x<=0 else ExactRat(x.numerator,x.denominator)
def sched_data(m):
    s=m.schedule
    if isinstance(s,ConstantSchedule): return ["constant",s.value.numerator,s.value.denominator]
    if isinstance(s,PiecewiseSchedule): return ["piecewise",[s.default.numerator,s.default.denominator],[[e,v.numerator,v.denominator] for e,v in s.points]]
    if isinstance(s,NamedSchedule): return ["named",resolve_named_schedule(s.schedule_id).schedule_id]
    raise CertificateGenerationError("unsupported schedule")
def digest(m):
    p={"n":m.n,"beliefs":[[q.numerator,q.denominator] for q in m.beliefs],"exposures":list(m.exposures),"threshold":[m.threshold.numerator,m.threshold.denominator],"schedule":sched_data(m)}
    return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:12]
def render_schedule(m):
    s=m.schedule
    if isinstance(s,ConstantSchedule): return f"fun _ => {rat(s.value)}"
    if isinstance(s,PiecewiseSchedule):
        b=rat(s.default)
        for e,v in reversed(s.points): b=f"if e = {e} then {rat(v)} else {b}"
        return f"fun e => {b}"
    if isinstance(s,NamedSchedule): return f"{resolve_named_schedule(s.schedule_id).lean_definition}.receptivityAt"
    raise CertificateGenerationError("unsupported schedule")
def render_state(m): return "!["+", ".join(f"⟨{rat(b)}, {e}⟩" for b,e in zip(m.beliefs,m.exposures,strict=True))+"]"
def header(): return [f"import {PROD}","","namespace NarrativeAnalyzerCertificate","","open NarrativeDynamics","open NarrativeDynamics.FitnessABMPathNExposure","open NarrativeDynamics.FitnessABMPathNExposureConvergence","open Filter Topology",""]
def defs(m,d):
    p=f"AnalyzerParams_{d}"; s=f"AnalyzerState_{d}"
    return [f"private def {p} : ExposureParameters :=",f"  ⟨{render_schedule(m)}, {rat(m.threshold)}⟩","",f"private def {s} : State {m.n} := {render_state(m)}",""],p,s
def ordered(xs,allowed):
    xs=tuple(xs)
    if len(xs)!=len(set(xs)) or any(x not in allowed for x in xs): raise ValueError("unsupported or duplicate certificate claim")
    return xs
def valid_lines(m,p,l):
    s=m.schedule; out=[f"private theorem {l} : {p}.Valid := by","  constructor"]
    if isinstance(s,ConstantSchedule): out += ["  · intro e",f"    norm_num [{p}]"]
    elif isinstance(s,PiecewiseSchedule): out += ["  · intro e",f"    simp only [{p}]","    split_ifs <;> norm_num"]
    elif isinstance(s,NamedSchedule): out += ["  · intro e",f"    exact ({VALID[resolve_named_schedule(s.schedule_id).schedule_id]}.1 e)"]
    else: raise CertificateGenerationError("unsupported schedule")
    return out+[f"  · norm_num [{p}]",""]
def broadcast_lines(p,s,l): return [f"private theorem {l} : allBroadcast {p} {s} := by","  intro i",f"  fin_cases i <;> norm_num [allBroadcast, {p}, {s}]",""]
def param_bad(m):
    if not 0<=F(m.threshold)<=1:return ("threshold",None)
    s=m.schedule
    if isinstance(s,NamedSchedule): resolve_named_schedule(s.schedule_id); return None
    pairs=((0,s.value),) if isinstance(s,ConstantSchedule) else (*s.points,)
    if isinstance(s,PiecewiseSchedule):
        used={e for e,_ in s.points}; w=next(i for i in range(len(used)+1) if i not in used); pairs=(*s.points,(w,s.default))
    for e,v in pairs:
        if not 0<=F(v)<=1:return ("schedule",e)
    return None
def broadcast_bad(m):
    t=F(m.threshold)
    for i,b in enumerate(m.beliefs):
        if not t<=F(b)<=1:return i
    return None

class CertificateBuilder:
    def build_structural_positive(self,m,claims):
        req=ordered(claims,STRUCT); d=digest(m); L=header(); ds,p,s=defs(m,d); L+=ds
        vl=f"AnalyzerParamsValid_{d}"; bl=f"AnalyzerAllBroadcast_{d}"; el=f"AnalyzerExposureLaw_{d}"; al=f"AnalyzerEffectiveAlphaLookup_{d}"
        if any(x in req for x in ("parameters_valid","exposure_law","effective_alpha_lookup")): L+=valid_lines(m,p,vl)
        if any(x in req for x in ("initial_all_broadcast","exposure_law","effective_alpha_lookup")): L+=broadcast_lines(p,s,bl)
        if "exposure_law" in req:L += [f"private theorem {el} (k : Nat) (i : Fin {m.n}) :",f"    (((step {p} {m.n})^[k] {s}) i).exposure = ({s} i).exposure + k * FitnessABMPathN.degree {m.n} i := by",f"  exact exposure_iterate {p} {vl} {m.n} (by norm_num) {s} {bl} k i",""]
        if "effective_alpha_lookup" in req:L += [f"private theorem {al} (k : Nat) (i : Fin {m.n}) :",f"    {p}.receptivityAt ((((step {p} {m.n})^[k] {s}) i).exposure + FitnessABMPathN.degree {m.n} i) =",f"      {p}.receptivityAt (({s} i).exposure + (k + 1) * FitnessABMPathN.degree {m.n} i) := by",f"  rw [exposure_iterate {p} {vl} {m.n} (by norm_num) {s} {bl} k i]","  congr 1","  simp only [Nat.add_mul, Nat.one_mul, Nat.add_assoc]",""]
        L += ["end NarrativeAnalyzerCertificate",""]
        meta={
          "parameters_valid":CertificateClaim("parameters_valid",ClaimStatus.PROVED,f"{vl}; definition {EXP}.ExposureParameters.Valid",(),{}),
          "initial_all_broadcast":CertificateClaim("initial_all_broadcast",ClaimStatus.PROVED,f"{bl}; definition {CONV}.allBroadcast",(),{}),
          "exposure_law":CertificateClaim("exposure_law",ClaimStatus.PROVED,f"{el}; {CONV}.exposure_iterate",("parameters_valid","initial_all_broadcast","n >= 2"),{}),
          "effective_alpha_lookup":CertificateClaim("effective_alpha_lookup",ClaimStatus.PROVED,f"{al}; {CONV}.exposure_iterate",("parameters_valid","initial_all_broadcast","n >= 2"),{"lookup":"e0(i) + (k+1) * degree(i)"})}
        return Certificate("\n".join(L),tuple(meta[x] for x in req))
    def build_structural_negative(self,m,claims):
        req=ordered(claims,NEG); d=digest(m); L=header(); ds,p,s=defs(m,d);L+=ds; out=[]
        for x in req:
            if x=="parameters_valid":
                w=param_bad(m)
                if w is None: raise ValueError("no parameter counter-witness")
                l=f"AnalyzerParamsInvalid_{d}"; kind,e=w; L += [f"private theorem {l} : ¬ {p}.Valid := by","  intro h"]
                L += ["  have hbad := h.2",f"  norm_num [{p}] at hbad"] if kind=="threshold" else [f"  have hbad := h.1 {e}",f"  norm_num [{p}] at hbad"]
                L += [""]; out.append(CertificateClaim(x,ClaimStatus.DISPROVED,f"{l}; negation of {EXP}.ExposureParameters.Valid",(),{}))
            else:
                i=broadcast_bad(m)
                if i is None: raise ValueError("no broadcast counter-witness")
                l=f"AnalyzerAllBroadcastInvalid_{d}"; L += [f"private theorem {l} : ¬ allBroadcast {p} {s} := by","  intro h",f"  have hbad := h ({i} : Fin {m.n})",f"  norm_num [allBroadcast, {p}, {s}] at hbad",""]
                out.append(CertificateClaim(x,ClaimStatus.DISPROVED,f"{l}; negation of {CONV}.allBroadcast",(),{"witness_index":str(i)}))
        L += ["end NarrativeAnalyzerCertificate",""]; return Certificate("\n".join(L),tuple(out))
    def build_pathn_consensus(self,m,eps):
        if not isinstance(m.schedule,(ConstantSchedule,PiecewiseSchedule)): raise CertificateGenerationError("global interior requires finite schedule")
        c=candidate_global_interior(m)
        if F(eps)<=0 or c is None or F(eps)>F(c): raise ValueError("invalid global-interior witness")
        d=digest(m);L=header();ds,p,s=defs(m,d);L+=ds;vl=f"AnalyzerParamsValid_{d}";bl=f"AnalyzerAllBroadcast_{d}";gl=f"AnalyzerGlobalInterior_{d}";rl=f"AnalyzerReachableInterior_{d}";cl=f"AnalyzerPathNConsensus_{d}";e=rat(eps)
        L+=valid_lines(m,p,vl)+broadcast_lines(p,s,bl)+[f"private theorem {gl} :",f"    ∀ e, {e} ≤ {p}.receptivityAt e ∧ {p}.receptivityAt e ≤ 1 - {e} := by","  intro e"]
        L += [f"  norm_num [{p}]",""] if isinstance(m.schedule,ConstantSchedule) else [f"  simp only [{p}]","  split_ifs <;> norm_num",""]
        L += [f"private theorem {rl} : ReachableInterior {p} {m.n} {s} {e} := by","  intro k i",f"  exact {gl} _","",f"private theorem {cl} :",f"    ∃ c : Real, ∀ i : Fin {m.n}, Tendsto (fun k => ((((step {p} {m.n})^[k] {s}) i).belief : Real)) atTop (nhds c) := by",f"  exact trajectory_consensus_exists_of_global_interior {p} {vl} {m.n} (by norm_num) {s} {bl} {e} (by norm_num) {gl}","","end NarrativeAnalyzerCertificate",""]
        ev={"eps":txt(eps)}
        return Certificate("\n".join(L),(CertificateClaim("reachable_interior",ClaimStatus.PROVED,f"{rl}; definition {CONV}.ReachableInterior",("global interior bound",),ev),CertificateClaim("pathn_consensus_exists",ClaimStatus.PROVED,f"{cl}; {CONV}.trajectory_consensus_exists_of_global_interior",("parameters_valid","n >= 2","initial_all_broadcast","global interior bound"),ev)))
    def build_named_path2(self,m,route:FixedFixtureRoute):
        actual=fixed_fixture_route(m)
        if actual is None or route!=actual or not isinstance(m.schedule,NamedSchedule): raise ValueError("named Path2 route mismatch")
        d=digest(m);L=header();ds,p,s=defs(m,d);L+=ds;pe=f"AnalyzerNamedParamsEq_{d}";se=f"AnalyzerNamedStateEq_{d}";pl=f"AnalyzerPath2Consensus_{d}";tail=route.theorem.rsplit('.',1)[-1]
        L += [f"private theorem {pe} : {p} = {route.schedule_id} := by","  rfl","",f"private theorem {se} : {s} = (![⟨1, 0⟩, ⟨0, 0⟩] : State 2) := by","  rfl","",*named_path2_body(schedule_id=route.schedule_id,params=p,state=s,params_eq=pe,state_eq=se,lemma=pl,theorem_tail=tail),"","end NarrativeAnalyzerCertificate",""]
        direct_provenance=f"{pl}; {route.theorem}"
        if route.schedule_id=="harmonicSchedule": return Certificate("\n".join(L),(CertificateClaim("path2_consensus",ClaimStatus.PROVED,direct_provenance,route.assumptions,{"value":"1/2"}),CertificateClaim("consensus_value_known",ClaimStatus.PROVED,direct_provenance,route.assumptions,{"value":"1/2"})))
        provenance=(f"{pl}; {CONV}.path2_mean_iterate; {route.theorem}" if route.schedule_id=="slowZeroSchedule" else direct_provenance)
        return Certificate("\n".join(L),(CertificateClaim("path2_consensus",ClaimStatus.DISPROVED,provenance,route.assumptions,{}),))
