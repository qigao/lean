import Browser.Interaction.Conformance

open Browser.Interaction

private def usage : String :=
  "usage: browser-interaction-trace-check <interaction.jsonl>"

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] =>
      let text ← IO.FS.readFile path
      match verifyInteractionJsonl text with
      | .ok _ =>
          IO.println s!"browser interaction journal: accepted {path}"
          return 0
      | .error message =>
          IO.eprintln s!"browser interaction journal rejected: {message}"
          return 1
  | _ =>
      IO.eprintln usage
      return 2
