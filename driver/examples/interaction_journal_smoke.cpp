#include "browser/interaction_journal.hpp"

#include <cassert>
#include <thread>
#include <vector>

namespace bi = browser::interaction;

class CollectSink final : public bi::InteractionSink {
 public:
  void write(const bi::InteractionRecord& record) override {
    records.push_back(record);
  }

  std::vector<bi::InteractionRecord> records;
};

int main() {
  CollectSink sink;
  bi::InteractionJournal journal(sink);

  journal.action_started(1, 2);
  journal.cdp_request(1, 2, 17);
  journal.action_started(2, 7);
  journal.input_dispatch(2, 7);

  assert(sink.records.size() == 4);
  assert(sink.records[0].seq == 1);
  assert(sink.records[1].seq == 2);
  assert(sink.records[2].seq == 3);
  assert(sink.records[3].seq == 4);
  assert(sink.records[0].lane == 1);
  assert(sink.records[2].lane == 2);
  assert(sink.records[1].kind == bi::InteractionKind::cdp_request);
  assert(sink.records[1].action == 2);
  assert(sink.records[1].call == 17);

  std::thread first([&journal] {
    for (int i = 0; i < 100; ++i) {
      journal.human_input(1);
    }
  });
  std::thread second([&journal] {
    for (int i = 0; i < 100; ++i) {
      journal.human_input(2);
    }
  });
  first.join();
  second.join();

  assert(sink.records.size() == 204);
  for (std::size_t i = 0; i < sink.records.size(); ++i) {
    assert(sink.records[i].seq == i + 1);
  }

  return 0;
}
