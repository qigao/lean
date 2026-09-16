import Lean.Data.Json
import NarrativeDynamics.Core.FitnessABMReplay
import NarrativeDynamics.Core.FitnessABMPathN

/-!
# Exact BB rational conformance checker

The corpus is finite and authored. JSON is parsed strictly enough to preserve
exact rational structure, while model results are recomputed through the
existing Lean propagation and fitness-replay implementations. This is finite
case conformance evidence, not a universal Python↔Lean equivalence theorem.
-/

namespace NarrativeDynamics.Conformance.BBRationalConformance

open Lean
open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment

private def schema : String := "bb-rational-conformance/v1"
private def generatorPath : String := "tools/generate_bb_rational_conformance.py"
private def checkerPath : String :=
  "NarrativeDynamics/Conformance/BBRationalConformance.lean"
private def contractHash : String :=
  "eb02d3e4cebc549458d3f1fdaf54a7bef5bb75587d1dc9b6bc81b168a3857924"

private def fail {α : Type} (message : String) : Except String α :=
  .error message

private def checkExactKeys
    (json : Json) (label : String) (expected : List String) : Except String Unit := do
  let object ← json.getObj?
  let actualCount := object.foldl (init := 0) (fun count _ _ => count + 1)
  if actualCount != expected.length then
    fail s!"{label}: expected exactly {expected.length} fields, got {actualCount}"
  for key in expected do
    if (object.get? key).isNone then
      fail s!"{label}: missing required field {key}"
  pure ()

private def checkString (json : Json) (field expected : String) : Except String Unit := do
  let actual ← (← json.getObjVal? field).getStr?
  if actual = expected then
    pure ()
  else
    fail s!"{field}: expected {expected}, got {actual}"

private def parseRat (json : Json) : Except String Rat := do
  checkExactKeys json "rational" ["num", "den"]
  let numerator ← (← json.getObjVal? "num").getInt?
  let denominator ← (← json.getObjVal? "den").getNat?
  if denominator = 0 then
    fail "rational denominator must be positive"
  else if numerator.natAbs.gcd denominator ≠ 1 then
    fail s!"non-reduced rational {numerator}/{denominator}"
  else
    pure ((numerator : Rat) / (denominator : Rat))

private def parseRatList (json : Json) : Except String (List Rat) := do
  let values ← json.getArr?
  values.toList.mapM parseRat

private def parseNatList (json : Json) : Except String (List Nat) := do
  let values ← json.getArr?
  values.toList.mapM Json.getNat?

private def parseBoolList (json : Json) : Except String (List Bool) := do
  let values ← json.getArr?
  values.toList.mapM Json.getBool?

private def parseEdge (json : Json) : Except String (Nat × Nat) := do
  let values ← json.getArr?
  match values.toList with
  | [left, right] =>
      pure (← left.getNat?, ← right.getNat?)
  | _ => fail "edge must contain exactly two node indices"

private def parseEdges (json : Json) : Except String (List (Nat × Nat)) := do
  let values ← json.getArr?
  values.toList.mapM parseEdge

private structure ParsedBirth where
  fitness : Rat
  targets : List Nat

private def parseBirth (json : Json) : Except String ParsedBirth := do
  pure {
    fitness := ← parseRat (← json.getObjVal? "fitness")
    targets := ← parseNatList (← json.getObjVal? "targets")
  }

private def parseBirths (json : Json) : Except String (List ParsedBirth) := do
  let values ← json.getArr?
  values.toList.mapM parseBirth

private def checkProvenance (json : Json) : Except String Unit := do
  checkExactKeys json "provenance"
    ["schema", "generator", "generator_contract", "lean_checker", "case_set"]
  checkString json "schema" schema
  checkString json "generator" generatorPath
  checkString json "generator_contract" contractHash
  checkString json "lean_checker" checkerPath
  checkString json "case_set" "v1"

private def expectNat (label : String) (actual expected : Nat) : Except String Unit :=
  if actual = expected then pure ()
  else fail s!"{label}: expected {expected}, got {actual}"

private def expectRats (label : String) (actual expected : List Rat) : Except String Unit :=
  if actual = expected then pure ()
  else fail s!"{label}: exact rational mismatch"

private def expectNats (label : String) (actual expected : List Nat) : Except String Unit :=
  if actual = expected then pure ()
  else fail s!"{label}: natural-number mismatch"

private def expectBools (label : String) (actual expected : List Bool) : Except String Unit :=
  if actual = expected then pure ()
  else fail s!"{label}: boolean mismatch"

private def expectEdges
    (label : String) (actual expected : List (Nat × Nat)) : Except String Unit :=
  if actual = expected then pure ()
  else fail s!"{label}: edge mismatch"

private def propagationResult2
    (alpha threshold : Rat) (beliefs : List Rat) (exposures : List Nat) :
    Except String (List Bool × List Rat × List Nat) := do
  match beliefs, exposures with
  | [b0, b1], [e0, e1] =>
      let population : Population 2 :=
        ⟨fun _ => ⟨alpha, threshold⟩, ![⟨b0, e0⟩, ⟨b1, e1⟩]⟩
      let next := propagate (NarrativeDynamics.FitnessABMPathN.pathAdj 2) population
      pure (
        List.ofFn fun i => broadcasting (population.profiles i) (population.agents i),
        List.ofFn fun i => (next.agents i).belief,
        List.ofFn fun i => (next.agents i).exposures)
  | _, _ => fail "two-node propagation case has invalid vector length"

private def propagationResult3
    (alpha threshold : Rat) (beliefs : List Rat) (exposures : List Nat) :
    Except String (List Bool × List Rat × List Nat) := do
  match beliefs, exposures with
  | [b0, b1, b2], [e0, e1, e2] =>
      let population : Population 3 :=
        ⟨fun _ => ⟨alpha, threshold⟩,
          ![⟨b0, e0⟩, ⟨b1, e1⟩, ⟨b2, e2⟩]⟩
      let next := propagate (NarrativeDynamics.FitnessABMPathN.pathAdj 3) population
      pure (
        List.ofFn fun i => broadcasting (population.profiles i) (population.agents i),
        List.ofFn fun i => (next.agents i).belief,
        List.ofFn fun i => (next.agents i).exposures)
  | _, _ => fail "three-node propagation case has invalid vector length"

private def checkPropagation
    (caseId : String) (input expected : Json) : Except String Unit := do
  let nodeCount ← (← input.getObjVal? "node_count").getNat?
  let edges ← parseEdges (← input.getObjVal? "edges")
  let alpha ← parseRat (← input.getObjVal? "alpha")
  let threshold ← parseRat (← input.getObjVal? "threshold")
  let beliefs ← parseRatList (← input.getObjVal? "beliefs")
  let exposures ← parseNatList (← input.getObjVal? "exposures")
  let expectedBroadcasting ← parseBoolList (← expected.getObjVal? "broadcasting")
  let expectedBeliefs ← parseRatList (← expected.getObjVal? "beliefs")
  let expectedExposures ← parseNatList (← expected.getObjVal? "exposures")

  let actual ←
    match caseId with
    | "inclusive-threshold" =>
        expectNat "node_count" nodeCount 2
        expectEdges "input.edges" edges [(0, 1)]
        expectRats "input.alpha/threshold" [alpha, threshold] [1/2, 1/2]
        expectRats "input.beliefs" beliefs [1/2, 0]
        expectNats "input.exposures" exposures [0, 0]
        propagationResult2 alpha threshold beliefs exposures
    | "no-broadcast" =>
        expectNat "node_count" nodeCount 2
        expectEdges "input.edges" edges [(0, 1)]
        expectRats "input.alpha/threshold" [alpha, threshold] [1/2, 1/2]
        expectRats "input.beliefs" beliefs [1/4, 1/8]
        expectNats "input.exposures" exposures [0, 0]
        propagationResult2 alpha threshold beliefs exposures
    | "nontrivial-mean" =>
        expectNat "node_count" nodeCount 3
        expectEdges "input.edges" edges [(0, 1), (1, 2)]
        expectRats "input.alpha/threshold" [alpha, threshold] [1/2, 1/2]
        expectRats "input.beliefs" beliefs [1, 1/2, 2/3]
        expectNats "input.exposures" exposures [0, 0, 0]
        propagationResult3 alpha threshold beliefs exposures
    | _ => fail s!"unknown propagation case {caseId}"

  expectBools "expected.broadcasting" actual.1 expectedBroadcasting
  expectRats "expected.beliefs" actual.2.1 expectedBeliefs
  expectNats "expected.exposures" actual.2.2 expectedExposures

private def rawSeedFrom (fitness : List Rat) (edges : List (Nat × Nat)) : RawSeed :=
  ⟨fitness.length, fitness.toArray, edges.toArray⟩

private def rawBirthFrom (birth : ParsedBirth) : RawBirth :=
  ⟨birth.fitness, birth.targets.toArray⟩

private def snapshotEdges {n : Nat} (snapshot : Snapshot n) : List (Nat × Nat) :=
  letI := snapshot.adjDec
  (List.ofFn fun i : Fin n => i).flatMap fun i =>
    (List.ofFn fun j : Fin n => j).filterMap fun j =>
      if i < j ∧ snapshot.graph.Adj i j then
        some (i.val, j.val)
      else
        none

private def recomputeBirthMasses
    (seed : RawSeed) (m : Nat) (births : List ParsedBirth) :
    Except String (List Rat) := do
  let initial ←
    match parseSeed seed with
    | .error error => fail s!"production seed parse failed: {reprStr error}"
    | .ok state => pure state
  match births with
  | [first, second] =>
      let firstStep ←
        match step initial m (rawBirthFrom first) with
        | .error error => fail s!"production first birth failed: {reprStr error}"
        | .ok out => pure out
      let secondStep ←
        match step firstStep.1 m (rawBirthFrom second) with
        | .error error => fail s!"production second birth failed: {reprStr error}"
        | .ok out => pure out
      pure [firstStep.2, secondStep.2]
  | _ => fail "success replay must contain exactly two births"

private def checkSuccessfulReplay (input expected : Json) : Except String Unit := do
  let m ← (← input.getObjVal? "m").getNat?
  let seedFitness ← parseRatList (← input.getObjVal? "seed_fitness")
  let seedEdges ← parseEdges (← input.getObjVal? "seed_edges")
  let births ← parseBirths (← input.getObjVal? "births")
  expectNat "input.m" m 1
  expectRats "input.seed_fitness" seedFitness [1, 1]
  expectEdges "input.seed_edges" seedEdges [(0, 1)]
  match births with
  | [first, second] =>
      expectRats "birth fitness" [first.fitness, second.fitness] [1, 1]
      expectNats "birth[0].targets" first.targets [1]
      expectNats "birth[1].targets" second.targets [2]
  | _ => fail "success replay must contain exactly two births"

  let expectedBirthMasses ← parseRatList (← expected.getObjVal? "birth_masses")
  let expectedTraceMass ← parseRat (← expected.getObjVal? "trace_mass")
  let expectedNodeCount ← (← expected.getObjVal? "node_count").getNat?
  let expectedBirthCount ← (← expected.getObjVal? "birth_count").getNat?
  let expectedTickCount ← (← expected.getObjVal? "tick_count").getNat?
  let expectedEdges ← parseEdges (← expected.getObjVal? "edges")
  expectNat "expected.birth_count" expectedBirthCount births.length
  expectNat "expected.tick_count" expectedTickCount births.length

  let seed := rawSeedFrom seedFitness seedEdges
  let actualBirthMasses ← recomputeBirthMasses seed m births
  expectRats "expected.birth_masses" actualBirthMasses expectedBirthMasses

  let rawBirths := births.map rawBirthFrom
  match FitnessAttachment.replay seed m rawBirths with
  | .error error => fail s!"production replay failed: {reprStr error}"
  | .ok result =>
      expectRats "expected.trace_mass" [result.probability] [expectedTraceMass]
      expectNat "expected.node_count" result.final.nodeCount expectedNodeCount
      expectEdges "expected.edges" (snapshotEdges result.final.state.snapshot) expectedEdges

private def checkDuplicateTarget (input expected : Json) : Except String Unit := do
  let m ← (← input.getObjVal? "m").getNat?
  let seedFitness ← parseRatList (← input.getObjVal? "seed_fitness")
  let seedEdges ← parseEdges (← input.getObjVal? "seed_edges")
  let birth ← parseBirth (← input.getObjVal? "birth")
  expectNat "input.m" m 2
  expectRats "input.seed_fitness" seedFitness [1, 1]
  expectEdges "input.seed_edges" seedEdges [(0, 1)]
  expectRats "input.birth.fitness" [birth.fitness] [1]
  expectNats "input.birth.targets" birth.targets [0, 0]
  checkString (← expected.getObjVal? "error") "stage" "tick_network"
  checkString (← expected.getObjVal? "error") "code" "duplicateTarget"
  checkString (← expected.getObjVal? "error") "field" "targets"

  let seed := rawSeedFrom seedFitness seedEdges
  match FitnessAttachment.replay seed m [rawBirthFrom birth] with
  | .error (.atBirth 0 .duplicateTarget) => pure ()
  | .error error => fail s!"unexpected production replay error: {reprStr error}"
  | .ok _ => fail "duplicate-target replay unexpectedly succeeded"

private def checkCase (json : Json) : Except String Unit := do
  checkExactKeys json "case"
    ["schema", "case_id", "kind", "provenance", "input", "expected"]
  checkString json "schema" schema
  let caseId ← (← json.getObjVal? "case_id").getStr?
  let kind ← (← json.getObjVal? "kind").getStr?
  checkProvenance (← json.getObjVal? "provenance")
  let input ← json.getObjVal? "input"
  let expected ← json.getObjVal? "expected"
  match kind with
  | "propagation" => checkPropagation caseId input expected
  | "replay" =>
      if caseId = "bb-successive-births" then
        checkSuccessfulReplay input expected
      else
        fail s!"unknown replay case {caseId}"
  | "error" =>
      if caseId = "duplicate-target-error" then
        checkDuplicateTarget input expected
      else
        fail s!"unknown error case {caseId}"
  | _ => fail s!"unknown case kind {kind}"

private def parseLines (contents : String) : Except String (List Json) :=
  ((contents.splitOn "\n").filter (fun line => line != "")).mapM Json.parse

def main (args : List String) : IO Unit := do
  match args with
  | [path] =>
      let contents ← IO.FS.readFile ⟨path⟩
      let cases ← IO.ofExcept (parseLines contents)
      for json in cases do
        IO.ofExcept (checkCase json)
      IO.println s!"{cases.length} cases checked"
  | _ =>
      throw (IO.userError
        "usage: BBRationalConformance <conformance/bb_rational_v1.jsonl>")

end NarrativeDynamics.Conformance.BBRationalConformance

def main (args : List String) : IO Unit :=
  NarrativeDynamics.Conformance.BBRationalConformance.main args
