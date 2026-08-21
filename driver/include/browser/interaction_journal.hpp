#pragma once

#include <cstdint>
#include <mutex>
#include <optional>

namespace browser::interaction {

using SequenceId = std::uint64_t;
using LaneId = std::uint64_t;
using ActionId = std::uint64_t;
using CdpCallId = std::uint64_t;
using Time = std::uint64_t;
using EventId = std::uint64_t;
using CorrelationId = std::uint64_t;
using PageId = std::uint64_t;

enum class InteractionKind : std::uint8_t {
  action_started,
  action_finished,
  cdp_request,
  cdp_response,
  deadline_armed,
  timer_expired,
  input_dispatch,
  human_input,
  policy_issued,
  policy_delivered,
  epoch_recreated,
  actor_destroyed,
};

struct InteractionRecord {
  SequenceId seq{};
  LaneId lane{};
  InteractionKind kind{};

  std::optional<ActionId> action;
  std::optional<CdpCallId> call;
  std::optional<Time> expires_at;
  std::optional<Time> now;
  std::optional<EventId> event_id;
  std::optional<CorrelationId> correlation_id;
  std::optional<std::uint32_t> depth;
  std::optional<PageId> page;
};

class InteractionSink {
 public:
  virtual ~InteractionSink() = default;
  virtual void write(const InteractionRecord& record) = 0;
};

/**
 * Thread-safe typed interaction journal.
 *
 * The journal owns total observation order: sequence assignment and sink
 * delivery happen under the same mutex. Driver components therefore never
 * construct JSON or allocate sequence numbers themselves.
 */
class InteractionJournal final {
 public:
  explicit InteractionJournal(InteractionSink& sink) noexcept;

  InteractionJournal(const InteractionJournal&) = delete;
  InteractionJournal& operator=(const InteractionJournal&) = delete;
  InteractionJournal(InteractionJournal&&) = delete;
  InteractionJournal& operator=(InteractionJournal&&) = delete;

  void action_started(LaneId lane, ActionId action);
  void action_finished(LaneId lane, ActionId action);
  void cdp_request(LaneId lane, ActionId action, CdpCallId call);
  void cdp_response(LaneId lane, ActionId action, CdpCallId call);
  void deadline_armed(LaneId lane, ActionId action, Time expires_at);
  void timer_expired(LaneId lane, ActionId action, Time now);
  void input_dispatch(LaneId lane, ActionId action);
  void human_input(LaneId lane);
  void policy_issued(
      LaneId lane,
      EventId event_id,
      CorrelationId correlation_id,
      std::uint32_t depth,
      PageId page);
  void policy_delivered(LaneId lane);
  void epoch_recreated(LaneId lane);
  void actor_destroyed(LaneId lane);

 private:
  void emit(InteractionRecord record);

  InteractionSink* sink_;
  SequenceId next_seq_{1};
  std::mutex mutex_;
};

}  // namespace browser::interaction
