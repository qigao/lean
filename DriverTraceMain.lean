import Browser.Conformance

open Browser

private def usage : String :=
  "usage: browser-driver-trace-check <trace.jsonl>"

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] =>
      let text ← IO.FS.readFile path
      match verifyJsonl { maxDepth := 32 } text with
      | .ok _ =>
          IO.println s!"browser-driver conformance: accepted {path}"
          return 0
      | .error message =>
          IO.eprintln s!"browser-driver conformance rejected: {message}"
          return 1
  | _ =>
      IO.eprintln usage
      return 2
