#include "browser/interaction_journal.hpp"
#include "browser/jsonl_interaction_sink.hpp"

#include <fstream>
#include <iostream>

namespace bi = browser::interaction;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: emit-interaction-trace <output.jsonl>\n";
    return 2;
  }

  std::ofstream output(argv[1]);
  if (!output) {
    std::cerr << "cannot open output file\n";
    return 2;
  }

  bi::JsonlInteractionSink sink(output);
  bi::InteractionJournal journal(sink);

  journal.action_started(1, 2);
  journal.action_started(2, 7);
  journal.deadline_armed(1, 2, 10);
  journal.timer_expired(1, 2, 10);
  journal.input_dispatch(2, 7);
  journal.cdp_request(2, 7, 31);
  journal.cdp_response(2, 7, 31);

  return 0;
}
