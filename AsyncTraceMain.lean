import Browser.AsyncConformance

open Browser

private def usage : String :=
  "usage: browser-async-trace-check <trace.jsonl>"

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] =>
      let text ← IO.FS.readFile path
      match verifyAsyncJsonl text with
      | .ok _ =>
          IO.println s!"browser async conformance: accepted {path}"
          return 0
      | .error message =>
          IO.println s!"browser async conformance rejected: {message}"
          return 1
  | _ =>
      IO.println usage
      return 2
