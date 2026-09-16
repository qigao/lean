import Lean.Data.Json
import NarrativeDynamics.Core.FitnessABMReplay

/-!
# Exact BB rational conformance checker

Task 4 RED establishes the real CLI/file/JSONL parsing boundary first. The
case checker is intentionally absent until this consumer fails at `checkCase`.
-/

namespace NarrativeDynamics.Conformance.BBRationalConformance

open Lean

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
