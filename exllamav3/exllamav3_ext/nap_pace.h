// Idle pacing for the host worker threads that spin-park on a counter: the MoE handoff worker and its
// stager (cpu/moe_handoff.cu), the MoE compute pool (cpu/moe_mul1.cpp) and the CPU all-reduce slice
// workers (parallel/all_reduce_cpu_avx2.cpp). All three pause-loop on a head/generation counter and,
// once the spin budget is spent, nap. Nothing wakes them on Linux - only the Windows pool has a wake
// path (WaitOnAddress/WakeByAddressAll) - so the nap length *is* the wake latency.
//
// Ladder: tier 1 while the queue has been dry for under 100 ms, tier 2 until 2 s, tier 3 afterwards.
// Both thresholds are measured from the instant the worker entered the nap regime, and every worker
// re-arms that instant: the idle counter is reset by each job and, in moe_handoff.cu, by each
// pass_wake. A live pass lands jobs per MoE layer, far less than 100 ms apart, so it can never climb
// past tier 1 - which makes tier 1 the only tier that can cost tok/s. 35192fc8 raised it from 50 to
// 100 us while chasing idle CPU, and that is exactly the regression it caused on decode paths whose
// per-layer gap exceeds the spin budget (~0.9 ms at 65536 pause iterations on Zen 4): the last waker
// of a barrier sleeps ~50 us longer than it has to, per layer. Default tier 1 back to 50 us and keep
// the deep tiers, which is inference-neutral against the pre-35192fc8 code (the ladder can only ever
// sleep longer than a flat 50 us after a *continuous* dry spell of 100 ms / 2 s, unreachable in-pass)
// and still collapses to 1 ms/10 ms naps between requests.
//
// Timer slack: Linux adds signal->timer_slack_ns (default 50 us) to every nanosleep, so on this box
// sleep_for(50 us) actually returns after ~104 us and sleep_for(100 us) after ~154 us. Dropping the
// slack to a couple of us on these threads makes a 50 us nap really ~55 us, i.e. better wake latency
// than either upstream tier-1, for ~3% -> ~5% of a core per parked thread. Slack is per-thread, so
// only threads that call this are affected; the inference thread keeps the kernel default.
//
// Knobs, read once per process (EXL3_NAP_TIER1_US=100 together with EXL3_TIMER_SLACK_NS=0 reproduces
// stock v1.6.7, which makes an A/B of all three arms a single build):
//   EXL3_NAP_TIER1_US       tier 1 nap, us      (default 50)
//   EXL3_NAP_TIER2_US       tier 2 nap, us      (default 1000)
//   EXL3_NAP_TIER3_US       tier 3 nap, us      (default 10000)
//   EXL3_TIMER_SLACK_NS     timer slack, ns     (default 2000; 0 leaves the kernel default alone)

#pragma once

#include <chrono>
#include <cstdlib>
#include <thread>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

namespace exl3_pace
{
    // Positive integer env knob, falling back to dflt on anything unparseable
    inline long env_num(const char* name, long dflt, long lo)
    {
        const char* s = getenv(name);
        if (!s || !s[0]) return dflt;
        char* end = nullptr;
        const long v = strtol(s, &end, 10);
        return (end != s && v >= lo) ? v : dflt;
    }

    inline long tier1_us() { static const long v = env_num("EXL3_NAP_TIER1_US", 50, 1); return v; }
    inline long tier2_us() { static const long v = env_num("EXL3_NAP_TIER2_US", 1000, 1); return v; }
    inline long tier3_us() { static const long v = env_num("EXL3_NAP_TIER3_US", 10000, 1); return v; }
    inline long slack_ns() { return env_num("EXL3_TIMER_SLACK_NS", 2000, 0); }

    // One nap of the ladder. dry_since is the instant this worker entered the nap regime, i.e. when
    // its idle counter last crossed the spin threshold; the caller re-stamps it at that moment.
    inline void nap(const std::chrono::steady_clock::time_point& dry_since)
    {
        const auto dry_for = std::chrono::steady_clock::now() - dry_since;
        const long us = (dry_for < std::chrono::milliseconds(100)) ? tier1_us()
                      : (dry_for < std::chrono::seconds(2)) ? tier2_us()
                      : tier3_us();
        std::this_thread::sleep_for(std::chrono::microseconds(us));
    }

    // Call once at the top of every napping worker: without it the kernel rounds a 50 us nap up to
    // ~104 us, so the whole first tier would be a rounding artifact.
    inline void apply_timer_slack()
    {
#if defined(__linux__)
        const long ns = slack_ns();
        static thread_local bool done = [&]() -> bool
        {
            if (ns > 0) prctl(PR_SET_TIMERSLACK, (unsigned long) ns, 0, 0, 0);
            return true;
        }();
        (void) done;
#else
        // No prctl on Windows/macOS. Windows is anyway not napping here (the 15.6 ms timer quantum
        // makes short sleeps useless, so those paths wait on WakeByAddress instead).
#endif
    }
}