"""
snllm.simulation.runner
~~~~~~~~~~~~~~~~~~~~~~~
Top-level simulation loop.

Two execution modes
-------------------
• run_sequential()   original single-thread path
• run_parallel(n_threads)  dialogue pairs executed concurrently
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import trange

from snllm.agents.core import Agent
from snllm.conversation.conversation import ConversationEngine
from snllm.environment.environment import NetworkEnvironment
from snllm.simulation.scheduler import Scheduler
from snllm.logging.models import StepLog, ConversationLog
from snllm.config.network import NetworkEnvironmentConfig, EmptyNetworkConfig


# ────────────────────────────────────────────────────────────────────
# helper type for thread returns
PairOutcome = Tuple[str, str, ConversationLog, float]  # uid, vid, log, new_strength
# ────────────────────────────────────────────────────────────────────


class SimulationRunner:
    """
    Executes successive steps:

        1. decay (memory & friendship)
        2. partner selection
        3. run ConversationEngine for each pair  ← can be parallel
        4. sync graph  ←→ agents
        5. snapshot & write StepLog
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        agents: List[Agent],
        *,
        graph_cfg: NetworkEnvironmentConfig | None = None,
        scheduler: Scheduler | None = None,
        max_steps: int = 50,
        log_dir: Path = Path("sim_logs"),
    ):
        self.agents: Dict[str, Agent] = {ag.id: ag for ag in agents}

        # friendship network
        self.env = NetworkEnvironment()
        self.env.initialize(self.agents.keys(), graph_cfg or EmptyNetworkConfig())

        # push initial strengths into every Agent’s FriendshipManager
        for u, v, s in self.env.edges():
            self.agents[u].friends.set_strength(v, strength=s, current_step=0)
            self.agents[v].friends.set_strength(u, strength=s, current_step=0)

        self.scheduler = scheduler or Scheduler()
        self.max_steps = max_steps
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._logs: List[StepLog] = []

    # ==================================================================
    # Public entry-points
    # ==================================================================
    def run(self) -> None:
        """Single-thread implementation"""
        self._run(loop_parallel=False)

    def run_parallel(self, *, n_threads: Optional[int] = None) -> None:
        """
        Execute each dialogue pair inside a ThreadPool.

        Parameters
        ----------
        n_threads :
            Pool size; set >= number of pairs for full overlap.
        """
        if n_threads is None:
            # one thread per pair, but cap at 256 to stay safe
            n_threads = min(len(self.agents) // 2, 256) or 1

        self._run(loop_parallel=True, n_threads=n_threads)

    # ==================================================================
    # Private core
    # ==================================================================
    def _run(self, *, loop_parallel: bool, n_threads: int = 4) -> None:

        for step in trange(self.max_steps, desc="Simulation"):
            # passive decay
            for ag in self.agents.values():
                ag.decay(step)

            # partner selection
            pairs = self.scheduler.pairs(self.agents, self.env, step=step)

            convo_logs: List[ConversationLog] = []

            # dialogue execution (possibly in parallel)
            if loop_parallel and pairs:
                with ThreadPoolExecutor(max_workers=n_threads) as pool:
                    futures = [
                        pool.submit(self._run_pair, uid, vid, step)
                        for uid, vid in pairs
                    ]
                    for fut in as_completed(futures):
                        uid, vid, clog, strength = fut.result()
                        convo_logs.append(clog)
                        # env sync in the main thread → race-free
                        self.env.update_edge(uid, vid, strength)
            else:  # sequential path
                for uid, vid in pairs:
                    uid, vid, clog, strength = self._run_pair(uid, vid, step)
                    convo_logs.append(clog)
                    self.env.update_edge(uid, vid, strength)

            # snapshot graph
            g_path = self.log_dir / f"graph_step_{step}.edgelist"
            self.env.snapshot(g_path)

            # persist StepLog
            self._logs.append(
                StepLog(step=step, conversations=convo_logs, graph_path=g_path)
            )
            self._save_step_json(self._logs[-1])

    # ------------------------------------------------------------------
    # Worker executed in threads
    # ------------------------------------------------------------------
    def _run_pair(self, uid: str, vid: str, step: int) -> PairOutcome:
        """
        Run a single conversation between agents *uid* and *vid*.

        Returns
        -------
        (uid, vid, ConversationLog, new_strength)
        """
        eng = ConversationEngine(self.agents[uid], self.agents[vid])
        eng.run(step)
        log = eng.log()

        strength = self.agents[uid].friends.strength(vid)
        return uid, vid, log, strength

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _save_step_json(self, slog: StepLog) -> None:
        path = self.log_dir / f"step_{slog.step}.json"
        with open(path, "w") as fh:
            fh.write(slog.model_dump_json())

    # ------------------------------------------------------------------
    # External access
    # ------------------------------------------------------------------
    def step_logs(self) -> List[StepLog]:  # immutable copy
        return list(self._logs)

    def graph(self) -> nx.Graph:  # read-only
        return self.env.as_networkx()
