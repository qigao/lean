#include "browser/interaction_boundary.hpp"

#include <cassert>
#include <stdexcept>
#include <vector>

namespace bi = browser::interaction;

class CollectSink final : public bi::InteractionSink {
 public:
  void write(const bi::InteractionRecord& record) override {
    records.push_back(record);
  }

  const bi::InteractionRecord& last() const {
    assert(!records.empty());
    return records.back();
  }

  std::vector<bi::InteractionRecord> records;
};

int main() {
  CollectSink sink;
  bi::InteractionJournal journal(sink);
  bi::InteractionBoundary boundary(journal);

  bool sent = false;
  const int result = boundary.cdp_request(1, 2, 17, [&] {
    assert(sink.last().kind == bi::InteractionKind::cdp_request);
    assert(sink.last().lane == 1);
    assert(sink.last().action == 2);
    assert(sink.last().call == 17);
    sent = true;
    return 42;
  });
  assert(sent);
  assert(result == 42);

  bool input_dispatched = false;
  boundary.input_dispatch(1, 2, [&] {
    assert(sink.last().kind == bi::InteractionKind::input_dispatch);
    assert(sink.last().action == 2);
    input_dispatched = true;
  });
  assert(input_dispatched);

  bool timeout_delivered = false;
  boundary.timer_expired(1, 2, 100, [&] {
    assert(sink.last().kind == bi::InteractionKind::timer_expired);
    assert(sink.last().now == 100);
    timeout_delivered = true;
  });
  assert(timeout_delivered);

  bool policy_enqueued = false;
  boundary.policy_issued(1, 50, 900, 0, 1, [&] {
    assert(sink.last().kind == bi::InteractionKind::policy_issued);
    assert(sink.last().event_id == 50);
    assert(sink.last().correlation_id == 900);
    assert(sink.last().page == 1);
    policy_enqueued = true;
  });
  assert(policy_enqueued);

  bool lifecycle_mutated = false;
  boundary.actor_destroyed(1, [&] {
    assert(sink.last().kind == bi::InteractionKind::actor_destroyed);
    lifecycle_mutated = true;
  });
  assert(lifecycle_mutated);

  const std::size_t before_throw = sink.records.size();
  bool threw = false;
  try {
    boundary.cdp_response(1, 2, 17, [] {
      throw std::runtime_error("response handler failed");
    });
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
  assert(sink.records.size() == before_throw + 1);
  assert(sink.last().kind == bi::InteractionKind::cdp_response);

  return 0;
}
