import Browser.Conformance

open Browser

private def usage : String :=
  "usage: browser-driver-trace-check <trace.jsonl>"

def main : IO Unit := do
  let args ← IO.getArgs
  match args with
  | [path] =>
      let text ← IO.FS.readFile path
      match verifyJsonl { maxDepth := 32 } text with
      | .ok _ =>
          IO.println s!"browser-driver conformance: accepted {path}"
      | .error message =>
          throw <| IO.userError s!"browser-driver conformance rejected: {message}"
  | _ => throw <| IO.userError usage
