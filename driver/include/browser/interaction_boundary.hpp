#pragma once

#include "browser/interaction_journal.hpp"

#include <functional>
#include <utility>

namespace browser::interaction {

/**
 * Thin instrumentation boundary for real Driver interactions.
 *
 * Every method records the typed interaction immediately before invoking the
 * supplied Driver callable. The journal record therefore means "this boundary
 * interaction was attempted/delivered"; it does not claim the callable later
 * succeeded. Exceptions and return values are propagated unchanged.
 *
 * This class intentionally owns no Browser/Page/Action state and no JSON
 * serialization. The real Driver remains responsible for lane assignment and
 * all domain behavior.
 */
class InteractionBoundary final {
 public:
  explicit InteractionBoundary(InteractionJournal& journal) noexcept : journal_(&journal) {}

  template <typename Fn>
  decltype(auto) action_started(LaneId lane, ActionId action, Fn&& transition) {
    journal_->action_started(lane, action);
    return std::invoke(std::forward<Fn>(transition));
  }

  template <typename Fn>
  decltype(auto) action_finished(LaneId lane, ActionId action, Fn&& transition) {
    journal_->action_finished(lane, action);
    return std::invoke(std::forward<Fn>(transition));
  }

  template <typename Fn>
  decltype(auto) cdp_request(LaneId lane, ActionId action, CdpCallId call, Fn&& send) {
    journal_->cdp_request(lane, action, call);
    return std::invoke(std::forward<Fn>(send));
  }

  template <typename Fn>
  decltype(auto) cdp_response(LaneId lane, ActionId action, CdpCallId call, Fn&& deliver) {
    journal_->cdp_response(lane, action, call);
    return std::invoke(std::forward<Fn>(deliver));
  }

  template <typename Fn>
  decltype(auto) deadline_armed(LaneId lane, ActionId action, Time expires_at, Fn&& arm) {
    journal_->deadline_armed(lane, action, expires_at);
    return std::invoke(std::forward<Fn>(arm));
  }

  template <typename Fn>
  decltype(auto) timer_expired(LaneId lane, ActionId action, Time now, Fn&& deliver) {
    journal_->timer_expired(lane, action, now);
    return std::invoke(std::forward<Fn>(deliver));
  }

  template <typename Fn>
  decltype(auto) input_dispatch(LaneId lane, ActionId action, Fn&& dispatch) {
    journal_->input_dispatch(lane, action);
    return std::invoke(std::forward<Fn>(dispatch));
  }

  template <typename Fn>
  decltype(auto) human_input(LaneId lane, Fn&& deliver) {
    journal_->human_input(lane);
    return std::invoke(std::forward<Fn>(deliver));
  }

  template <typename Fn>
  decltype(auto) policy_issued(
      LaneId lane,
      EventId event_id,
      CorrelationId correlation_id,
      std::uint32_t depth,
      PageId page,
      Fn&& enqueue) {
    journal_->policy_issued(lane, event_id, correlation_id, depth, page);
    return std::invoke(std::forward<Fn>(enqueue));
  }

  template <typename Fn>
  decltype(auto) policy_delivered(LaneId lane, Fn&& deliver) {
    journal_->policy_delivered(lane);
    return std::invoke(std::forward<Fn>(deliver));
  }

  template <typename Fn>
  decltype(auto) epoch_recreated(LaneId lane, Fn&& mutate) {
    journal_->epoch_recreated(lane);
    return std::invoke(std::forward<Fn>(mutate));
  }

  template <typename Fn>
  decltype(auto) actor_destroyed(LaneId lane, Fn&& mutate) {
    journal_->actor_destroyed(lane);
    return std::invoke(std::forward<Fn>(mutate));
  }

 private:
  InteractionJournal* journal_;
};

}  // namespace browser::interaction
