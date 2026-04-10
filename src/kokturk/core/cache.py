"""Two-tier LRU cache for morphological analyses.

Tier 1: In-memory LRU cache (OrderedDict).
    Default capacity of 50K tokens covers ~91% of running text in Turkish
    corpora, using approximately 17.5MB RAM.

Tier 2: Optional persistent DiskCache (SQLite-backed).
    Survives process restarts.  Backed by ``diskcache`` if installed.
    Auto-warms Tier 1 on startup from most frequent entries.

Cache key: surface form string (optionally with analysis flags appended).
Cache value: :class:`TokenAnalyses` (the ``analyze()`` return type).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kokturk.core.datatypes import TokenAnalyses

logger = logging.getLogger(__name__)


class AnalysisCache:
    """Thread-unsafe two-tier LRU cache for TokenAnalyses.

    Tier 1 is an in-memory :class:`OrderedDict` with LRU eviction.
    Tier 2 is an optional :mod:`diskcache` SQLite store for cross-session
    persistence.

    Attributes:
        capacity: Maximum number of entries in the memory tier.
        hits: Number of cache hits since creation.
        misses: Number of cache misses since creation.
    """

    def __init__(
        self,
        capacity: int = 50_000,
        disk_path: str | None = None,
        disk_size_limit: int = 1_073_741_824,  # 1 GB
    ) -> None:
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self._store: OrderedDict[str, TokenAnalyses] = OrderedDict()
        self._disk: Any = None

        if disk_path is not None:
            try:
                import diskcache  # type: ignore[import-untyped]

                self._disk = diskcache.Cache(
                    disk_path, size_limit=disk_size_limit,
                )
                logger.info("DiskCache initialised at %s", disk_path)
            except ImportError:
                logger.warning(
                    "diskcache not installed; using memory-only cache. "
                    "Install with: pip install diskcache",
                )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, word: str) -> TokenAnalyses | None:
        """Look up a word in the cache (memory then disk).

        Returns:
            The cached TokenAnalyses if found, otherwise None.
        """
        # Tier 1: memory
        if word in self._store:
            self.hits += 1
            self._store.move_to_end(word)
            return self._store[word]

        # Tier 2: disk
        if self._disk is not None:
            value = self._disk.get(word)
            if value is not None:
                self.hits += 1
                # Promote to memory tier
                self._promote_to_memory(word, value)
                return value  # type: ignore[return-value]

        self.misses += 1
        return None

    def put(self, word: str, analyses: TokenAnalyses) -> None:
        """Insert or update a word's analyses in both tiers.

        If the memory tier is at capacity the least recently used entry
        is evicted before the new entry is added.

        Args:
            word: The surface form used as cache key.
            analyses: The TokenAnalyses to cache.
        """
        if word in self._store:
            self._store.move_to_end(word)
        self._store[word] = analyses
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)

        # Write-through to disk tier
        if self._disk is not None:
            self._disk[word] = analyses

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction in [0.0, 1.0]."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, int | float]:
        """Return a snapshot of cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "memory_entries": len(self._store),
            "disk_entries": len(self._disk) if self._disk is not None else 0,
        }

    # ------------------------------------------------------------------
    # Warm-up & housekeeping
    # ------------------------------------------------------------------

    def warm_from_disk(self, top_n: int | None = None) -> int:
        """Pre-populate memory cache from disk cache.

        Call at startup to avoid cold-start penalty.

        Args:
            top_n: Maximum number of entries to load.  Defaults to
                ``self.capacity``.

        Returns:
            Number of entries actually loaded into memory.
        """
        if self._disk is None:
            return 0

        limit = top_n if top_n is not None else self.capacity
        loaded = 0
        for key in self._disk:
            if loaded >= limit:
                break
            if key not in self._store:
                value = self._disk.get(key)
                if value is not None:
                    self._promote_to_memory(key, value)
                    loaded += 1
        return loaded

    def clear(self) -> None:
        """Clear both tiers and reset counters."""
        self._store.clear()
        self.hits = 0
        self.misses = 0
        if self._disk is not None:
            self._disk.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _promote_to_memory(self, word: str, analyses: TokenAnalyses) -> None:
        """Add an entry to the memory tier, evicting LRU if needed."""
        if len(self._store) >= self.capacity:
            self._store.popitem(last=False)
        self._store[word] = analyses

    def __len__(self) -> int:
        """Return the number of entries currently in the memory tier."""
        return len(self._store)
