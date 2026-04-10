"""MorphoAnalyzer — unified interface to morphological analysis backends.

Supports Zeyrek (Python port of Zemberek) and TRMorph (foma FST) as
independent backends. Having two backends is critical for the CONF
(conflict density) component of the active learning acquisition function.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from kokturk.core.cache import AnalysisCache
from kokturk.core.compound_lexicon import decompose_fused_lvc
from kokturk.core.constants import TRMORPH_TO_CANONICAL, ZEYREK_TO_CANONICAL
from kokturk.core.datatypes import Morpheme, MorphologicalAnalysis, TokenAnalyses

logger = logging.getLogger(__name__)


class AnalyzerBackend(ABC):
    """Abstract base for morphological analysis backends."""

    @abstractmethod
    def analyze(self, word: str) -> list[MorphologicalAnalysis]:
        """Return all candidate parses for a word."""
        ...

    def close(self) -> None:  # noqa: B027
        """Release any resources held by this backend."""


class ZeyrekBackend(AnalyzerBackend):
    """Backend using Zeyrek (Python port of Zemberek).

    Zeyrek provides morphological analysis via a pure-Python implementation.
    It is the primary backend — always available without external dependencies.

    Zeyrek's analyze() returns list[list[Parse]], where Parse is a namedtuple
    with fields: word, lemma, pos, morphemes (list[str]), formatted.
    """

    def __init__(self) -> None:
        try:
            import zeyrek

            self._analyzer = zeyrek.MorphAnalyzer()
        except ImportError as e:
            raise ImportError(
                "Zeyrek is required: pip install zeyrek"
            ) from e

    def analyze(self, word: str) -> list[MorphologicalAnalysis]:
        """Analyze a word using Zeyrek and convert to canonical format."""
        try:
            raw_results: list[Any] = self._analyzer.analyze(word)
        except Exception:
            logger.warning("Zeyrek failed on word: %s", word, exc_info=True)
            return []

        # raw_results is list[list[Parse]] — flatten the inner lists
        analyses: list[MorphologicalAnalysis] = []
        for word_parses in raw_results:
            for parse in word_parses:
                root, tags, morphemes = _convert_zeyrek_parse(word, parse)
                total_parses = sum(len(wp) for wp in raw_results)
                analysis = MorphologicalAnalysis(
                    surface=word,
                    root=root,
                    tags=tuple(tags),
                    morphemes=tuple(morphemes),
                    source="zeyrek",
                    score=1.0 / max(total_parses, 1),
                )
                analyses.append(analysis)
        return analyses


# Derivational morpheme IDs in Zeyrek notation
_DERIVATIONAL_MORPHEMES: frozenset[str] = frozenset({
    "Become", "Acquire", "Dim", "Agt", "Ness", "With", "Without",
    "Related", "FitFor", "Ly", "Inf1", "Inf2", "Inf3",
    "PastPart", "FutPart", "PresPart", "NarrPart", "AorPart",
    "Caus", "Pass", "Recip", "Reflex",
})


def _convert_zeyrek_parse(
    surface: str,
    parse: Any,
) -> tuple[str, list[str], list[Morpheme]]:
    """Convert a single Zeyrek Parse namedtuple to canonical format.

    Zeyrek Parse has: word, lemma, pos, morphemes (list[str]), formatted.
    Example morphemes: ['Noun', 'A3pl', 'P3sg', 'Abl']

    Args:
        surface: The original word form.
        parse: A Zeyrek Parse namedtuple.

    Returns:
        Tuple of (root, canonical_tags, morphemes).
    """
    root = parse.lemma if hasattr(parse, "lemma") else surface

    tags: list[str] = []
    morphemes: list[Morpheme] = []

    # parse.morphemes is a list of strings like ['Noun', 'A3pl', 'P3sg', 'Abl']
    raw_morphemes: list[str] = parse.morphemes if hasattr(parse, "morphemes") else []

    for morph_id in raw_morphemes:
        canonical = ZEYREK_TO_CANONICAL.get(morph_id, "")
        if canonical:
            tags.append(canonical)

        category = "derivational" if morph_id in _DERIVATIONAL_MORPHEMES else "inflectional"

        morphemes.append(Morpheme(
            surface=morph_id,
            canonical=canonical if canonical else morph_id,
            category=category,
        ))

    return root, tags, morphemes


class TRMorphBackend(AnalyzerBackend):
    """Backend using TRMorph (foma FST) via flookup in pipe mode.

    Uses a long-lived subprocess to avoid per-word spawn overhead.
    Line-buffered with deadlock prevention measures.
    """

    def __init__(self, fst_path: str | None = None) -> None:
        if fst_path is None:
            fst_path = os.environ.get(
                "TRMORPH_FST_PATH", "./tools/trmorph/trmorph.fst"
            )

        if not os.path.exists(fst_path):
            raise FileNotFoundError(
                f"TRMorph FST not found at {fst_path}. "
                "Clone: git clone --branch trmorph2 --depth 1 "
                "https://github.com/coltekin/TRmorph.git tools/trmorph"
            )

        if not shutil.which("flookup"):
            raise FileNotFoundError(
                "flookup not found. Install foma-bin: apt install foma-bin"
            )

        self._fst_path = fst_path
        self._proc = subprocess.Popen(
            ["flookup", "-b", fst_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # Line-buffered to prevent deadlock
        )

    def analyze(self, word: str) -> list[MorphologicalAnalysis]:
        """Analyze a word using TRMorph FST via flookup pipe."""
        if self._proc.stdin is None or self._proc.stdout is None:
            return []

        try:
            self._proc.stdin.write(word + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError:
            logger.error("TRMorph flookup process died unexpectedly")
            return []

        analyses: list[MorphologicalAnalysis] = []
        while True:
            line = self._proc.stdout.readline().strip()
            if not line:
                break

            # flookup output format: "surface\tanalysis" or "surface\t+?"
            parts = line.split("\t")
            if len(parts) < 2 or parts[1] == "+?":
                continue

            root, tags, morphemes = _parse_trmorph_output(word, parts[1])
            analysis = MorphologicalAnalysis(
                surface=word,
                root=root,
                tags=tuple(tags),
                morphemes=tuple(morphemes),
                source="trmorph",
                score=1.0,
            )
            analyses.append(analysis)

        return analyses

    def close(self) -> None:
        """Terminate the flookup subprocess."""
        if self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=5)

    def __enter__(self) -> TRMorphBackend:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _parse_trmorph_output(
    surface: str, analysis_str: str
) -> tuple[str, list[str], list[Morpheme]]:
    """Parse a TRMorph FST output string into canonical format.

    TRMorph output format: ``root<tag1><tag2>...``
    Example: ``ev<N><pl><p3s><abl>``

    Args:
        surface: The original word form (fallback if root cannot be parsed).
        analysis_str: Raw flookup output string for one analysis.

    Returns:
        Tuple of (root, canonical_tags, morphemes).
    """
    # Split on < to get root and tags
    parts = analysis_str.replace(">", "").split("<")
    root = parts[0] if parts else surface
    raw_tags = parts[1:] if len(parts) > 1 else []

    tags: list[str] = []
    morphemes: list[Morpheme] = []

    for raw_tag in raw_tags:
        canonical = TRMORPH_TO_CANONICAL.get(raw_tag, "")
        if canonical:
            tags.append(canonical)
        morphemes.append(Morpheme(
            surface=raw_tag,
            canonical=canonical if canonical else raw_tag,
            category="inflectional",
        ))

    return root, tags, morphemes


class NeuralBackend(AnalyzerBackend):
    """Backend using trained GRU Seq2Seq morphological atomizer.

    Loads a trained model checkpoint and runs greedy decode to
    produce morphological analyses.

    Args:
        model_path: Path to the saved model checkpoint (.pt file).
        vocab_dir: Directory containing char_vocab.json and tag_vocab.json.
    """

    def __init__(
        self,
        model_path: str = "models/atomizer_v2/best_model.pt",
        vocab_dir: str = "models/vocabs",
    ) -> None:
        # Load vocabs
        from pathlib import Path as _Path

        import torch

        from kokturk.models.char_gru import MorphAtomizer
        from train.datasets import Vocab

        self._char_vocab = Vocab.load(_Path(f"{vocab_dir}/char_vocab.json"))
        self._tag_vocab = Vocab.load(_Path(f"{vocab_dir}/tag_vocab.json"))

        # Load model
        ckpt = torch.load(model_path, weights_only=True, map_location="cpu")
        self._model = MorphAtomizer(
            char_vocab_size=ckpt["char_vocab_size"],
            tag_vocab_size=ckpt["tag_vocab_size"],
            embed_dim=ckpt["embed_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
        )
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        self._torch = torch

    def analyze(self, word: str) -> list[MorphologicalAnalysis]:
        """Analyze a word using the trained neural model."""
        from train.datasets import EOS_IDX, PAD_IDX

        # Encode characters
        char_ids = [self._char_vocab.encode(c) for c in word]
        char_ids.append(EOS_IDX)
        char_ids = char_ids[:64]
        char_ids += [PAD_IDX] * (64 - len(char_ids))
        chars = self._torch.tensor([char_ids], dtype=self._torch.long)

        # Greedy decode
        preds = self._model.greedy_decode(chars)

        # Decode tag sequence
        tags: list[str] = []
        root = word
        for i, idx in enumerate(preds[0].tolist()):
            if idx == EOS_IDX:
                break
            if idx <= 3:  # PAD/SOS/EOS/UNK
                continue
            token = self._tag_vocab.decode(idx)
            if i == 0 and not token.startswith("+"):
                root = token
            else:
                tags.append(token)

        morphemes = tuple(
            Morpheme(surface=t, canonical=t, category="inflectional")
            for t in tags
        )
        return [MorphologicalAnalysis(
            surface=word,
            root=root,
            tags=tuple(tags),
            morphemes=morphemes,
            source="neural",
            score=1.0,
        )]


class DisambiguatorBackend(AnalyzerBackend):
    """Backend using BERTurk + Zeyrek candidate disambiguation.

    Wraps a trained :class:`~kokturk.models.disambiguator.BERTurkDisambiguator`
    that selects the best parse from Zeyrek's candidate list using sentence
    context from frozen BERTurk embeddings.

    For single-word analysis (no sentence context), returns all Zeyrek
    candidates. Sentence-level disambiguation is available via
    :meth:`analyze_in_context`.
    """

    def __init__(
        self,
        model_path: str = "models/v6/disambiguator/best_model.pt",
        bert_path: str = "models/berturk",
        vocab_dir: str = "models/vocabs",
    ) -> None:
        import json
        import torch
        from kokturk.models.disambiguator import BERTurkDisambiguator
        from train.datasets import Vocab

        self._zeyrek = ZeyrekBackend()

        tag_vocab = Vocab.load(Path(vocab_dir) / "tag_vocab.json")
        self._tag_vocab = tag_vocab

        ckpt = torch.load(model_path, map_location="cpu")
        tag_vocab_size = ckpt.get("tag_vocab_size", len(tag_vocab))
        self._model = BERTurkDisambiguator(
            tag_vocab_size=tag_vocab_size,
            bert_path=bert_path,
        )
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        logger.info(
            "DisambiguatorBackend loaded from %s (val EM=%.1f%%)",
            model_path, ckpt.get("val_em", 0) * 100,
        )

    def analyze(self, word: str) -> list[MorphologicalAnalysis]:
        """Single-word analysis without context — returns Zeyrek candidates."""
        return self._zeyrek.analyze(word)

    def analyze_in_context(
        self,
        word: str,
        sentence: str,
        position: int,
    ) -> MorphologicalAnalysis | None:
        """Disambiguate a word using sentence context.

        Args:
            word: Surface form to analyze.
            sentence: Full sentence containing the word.
            position: 0-based word position in the sentence.

        Returns:
            The highest-scoring analysis, or None if no candidates.
        """
        import torch

        candidates = self._zeyrek.analyze(word)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # Encode candidates
        max_parse_len = 15
        K = len(candidates)
        cand_ids = torch.zeros(1, K, max_parse_len, dtype=torch.long)
        cand_mask = torch.zeros(1, K, dtype=torch.bool)

        for i, cand in enumerate(candidates):
            parts = cand.to_str().split()
            for j, part in enumerate(parts[:max_parse_len]):
                cand_ids[0, i, j] = self._tag_vocab.encode(part)
            cand_mask[0, i] = True

        # Pad to at least 10 candidates (model expects fixed K)
        if K < 10:
            padded_ids = torch.zeros(1, 10, max_parse_len, dtype=torch.long)
            padded_mask = torch.zeros(1, 10, dtype=torch.bool)
            padded_ids[0, :K] = cand_ids[0, :K]
            padded_mask[0, :K] = cand_mask[0, :K]
            cand_ids = padded_ids
            cand_mask = padded_mask

        with torch.no_grad():
            logits, _ = self._model(
                sentence_texts=[sentence],
                target_positions=torch.tensor([position]),
                candidate_ids=cand_ids,
                candidate_mask=cand_mask,
            )

        best_idx = logits[0, :K].argmax().item()
        return candidates[best_idx]


# Registry of available backends
_BACKEND_REGISTRY: dict[str, type[AnalyzerBackend]] = {
    "zeyrek": ZeyrekBackend,
    "trmorph": TRMorphBackend,
    "neural": NeuralBackend,
    "disambiguator": DisambiguatorBackend,
}


class MorphoAnalyzer:
    """Unified morphological analyzer with multi-backend support and caching.

    Args:
        backends: List of backend names to use. Default: ["zeyrek"].
            Available: "zeyrek", "trmorph", "neural".
        cache_capacity: LRU cache capacity. Default: 50,000 tokens.
        model_path: Path to neural model checkpoint (for "neural" backend).
        vocab_dir: Path to vocabulary directory (for "neural" backend).
    """

    def __init__(
        self,
        backends: list[str] | None = None,
        cache_capacity: int = 50_000,
        model_path: str = "models/atomizer_v2/best_model.pt",
        vocab_dir: str = "models/vocabs",
    ) -> None:
        if backends is None:
            backends = ["zeyrek"]

        self._backends: list[AnalyzerBackend] = []
        for name in backends:
            if name not in _BACKEND_REGISTRY:
                raise ValueError(
                    f"Unknown backend: {name}. Available: {list(_BACKEND_REGISTRY)}"
                )
            try:
                if name == "neural":
                    backend = _BACKEND_REGISTRY[name](  # type: ignore[call-arg]
                        model_path=model_path, vocab_dir=vocab_dir,
                    )
                elif name == "disambiguator":
                    backend = _BACKEND_REGISTRY[name](  # type: ignore[call-arg]
                        vocab_dir=vocab_dir,
                    )
                else:
                    backend = _BACKEND_REGISTRY[name]()
                self._backends.append(backend)
                logger.info("Initialized backend: %s", name)
            except (ImportError, FileNotFoundError) as e:
                logger.warning("Backend %s unavailable: %s", name, e)

        if not self._backends:
            raise RuntimeError("No backends could be initialized")

        self._cache = AnalysisCache(capacity=cache_capacity)

    def analyze(
        self,
        word: str,
        *,
        decompose_lvc: bool = False,
        handle_special_tokens: bool = False,
        handle_code_switch: bool = False,
    ) -> TokenAnalyses:
        """Analyze a single word using all backends.

        Results are deduplicated by (root, tags) parse identity.
        Cached for subsequent lookups.

        Args:
            word: A single Turkish word to analyze.
            decompose_lvc: If True, check the surface form against the
                fused-LVC lexicon (:mod:`kokturk.core.compound_lexicon`)
                BEFORE running the standard backends. When a match is
                found, the analysis is rewritten so that the underlying
                nominal becomes the root and a ``+LVC.ET`` / ``+LVC.OL``
                tag is prepended to the suffix sequence. Default ``False``
                preserves bit-for-bit backward compatibility.
            handle_special_tokens: If True, route abbreviations, numerics,
                and reduplicated forms through the special-token
                preprocessor (:mod:`kokturk.core.special_tokens`) before
                analysis. Default ``False`` — opt-in to avoid silent
                corruption from preprocessor false positives on normal
                tokens.
            handle_code_switch: If True, detect foreign roots with Turkish
                suffixes separated by an apostrophe (e.g. ``Google'ladım``
                → ``Google +FOREIGN +VERB.LA +PAST +1SG``). Uses relaxed
                vowel harmony. Default ``False`` for backward compat.

        Returns:
            TokenAnalyses containing all unique parses from all backends.
        """
        # Cache key incorporates the option flags so that opt-in pipelines
        # do not collide with default pipelines.
        cache_key = word
        if decompose_lvc or handle_special_tokens or handle_code_switch:
            cache_key = (
                f"{word}\0lvc={int(decompose_lvc)}"
                f"\0sp={int(handle_special_tokens)}"
                f"\0cs={int(handle_code_switch)}"
            )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if handle_code_switch:
            from kokturk.core.code_switch import detect_code_switch

            cs_result = detect_code_switch(word)
            if cs_result is not None:
                result = self._analyze_code_switch(word, cs_result)
                self._cache.put(cache_key, result)
                return result

        if handle_special_tokens:
            from kokturk.core.special_tokens import preprocess_special_token

            special = preprocess_special_token(word)
            if special is not None:
                result = self._analyze_special(word, special)
                self._cache.put(cache_key, result)
                return result

        all_analyses: list[MorphologicalAnalysis] = []
        seen_parses: set[tuple[str, tuple[str, ...]]] = set()

        for backend in self._backends:
            for analysis in backend.analyze(word):
                identity = analysis.parse_identity()
                if identity not in seen_parses:
                    seen_parses.add(identity)
                    all_analyses.append(analysis)

        if decompose_lvc:
            decomposition = decompose_fused_lvc(word)
            if decomposition is not None:
                nominal, light_verb, _remainder = decomposition
                lvc_tag = f"+LVC.{light_verb.upper()}"
                rewritten: list[MorphologicalAnalysis] = []
                seen_lvc: set[tuple[str, tuple[str, ...]]] = set()
                for analysis in all_analyses:
                    new_tags = (lvc_tag,) + analysis.tags
                    new_morphemes = (
                        Morpheme(
                            surface=light_verb,
                            canonical=lvc_tag,
                            category="derivational",
                        ),
                    ) + analysis.morphemes
                    rewritten_analysis = MorphologicalAnalysis(
                        surface=word,
                        root=nominal,
                        tags=new_tags,
                        morphemes=new_morphemes,
                        source=analysis.source + "+lvc",
                        score=analysis.score,
                    )
                    identity = rewritten_analysis.parse_identity()
                    if identity not in seen_lvc:
                        seen_lvc.add(identity)
                        rewritten.append(rewritten_analysis)
                all_analyses = rewritten

        result = TokenAnalyses(
            surface=word,
            analyses=tuple(all_analyses),
        )
        self._cache.put(cache_key, result)
        return result

    def _analyze_special(
        self, word: str, special: object
    ) -> TokenAnalyses:
        """Build a TokenAnalyses for a special-token preprocessor result.

        The base form is taken as the root, the special-token type is
        emitted as a leading derivational tag, and any suffix part is
        analyzed as a single derived morpheme. This is intentionally
        coarse: the goal is to expose the special-token structure to
        downstream models, not to fully parse the suffix.
        """
        from kokturk.core.special_token_types import SpecialTokenResult

        assert isinstance(special, SpecialTokenResult)
        type_tag = f"+{special.token_type.upper()}"
        tags: list[str] = [type_tag]
        morphemes: list[Morpheme] = [
            Morpheme(
                surface=special.base,
                canonical=type_tag,
                category="derivational",
            )
        ]
        if special.suffix_part:
            tags.append("+SFX")
            morphemes.append(
                Morpheme(
                    surface=special.suffix_part,
                    canonical="+SFX",
                    category="inflectional",
                )
            )
        analysis = MorphologicalAnalysis(
            surface=word,
            root=special.base,
            tags=tuple(tags),
            morphemes=tuple(morphemes),
            source="special",
            score=1.0,
        )
        return TokenAnalyses(surface=word, analyses=(analysis,))

    def _analyze_code_switch(
        self, word: str, cs_result: object
    ) -> TokenAnalyses:
        """Build a TokenAnalyses for a code-switched token.

        The foreign root is taken as the root with a ``+FOREIGN`` leading
        tag, and the Turkish suffix chain is decomposed via regex-based
        pattern matching with relaxed vowel harmony.
        """
        from kokturk.core.code_switch import (
            CodeSwitchResult,
            analyze_foreign_suffixes,
        )
        from kokturk.core.phonology import last_vowel

        assert isinstance(cs_result, CodeSwitchResult)

        root_vowel = last_vowel(cs_result.foreign_root)
        suffix_tags = analyze_foreign_suffixes(
            cs_result.suffix_part, root_vowel, relaxed_harmony=True
        )

        tags: list[str] = ["+FOREIGN"]
        morphemes: list[Morpheme] = [
            Morpheme(
                surface=cs_result.foreign_root,
                canonical="+FOREIGN",
                category="derivational",
            )
        ]

        for tag in suffix_tags:
            morphemes.append(
                Morpheme(
                    surface=tag,
                    canonical=tag,
                    category="inflectional",
                )
            )
        tags.extend(suffix_tags)

        analysis = MorphologicalAnalysis(
            surface=word,
            root=cs_result.foreign_root,
            tags=tuple(tags),
            morphemes=tuple(morphemes),
            source="code_switch",
            score=0.9,
        )
        return TokenAnalyses(surface=word, analyses=(analysis,))

    def analyze_sentence(
        self,
        sentence: str,
    ) -> list[TokenAnalyses]:
        """Analyze all words in a sentence with contextual disambiguation.

        If a :class:`DisambiguatorBackend` is available, uses BERTurk
        sentence context to select the best parse for ambiguous tokens.
        Otherwise, falls back to per-word :meth:`analyze`.

        Args:
            sentence: A Turkish sentence (space-separated words).

        Returns:
            List of TokenAnalyses, one per word.
        """
        words = sentence.split()
        # Check if any backend supports sentence-level disambiguation
        disambiguator = None
        for backend in self._backends:
            if isinstance(backend, DisambiguatorBackend):
                disambiguator = backend
                break

        if disambiguator is None:
            return [self.analyze(w) for w in words]

        results: list[TokenAnalyses] = []
        for i, word in enumerate(words):
            best = disambiguator.analyze_in_context(word, sentence, i)
            if best is not None:
                results.append(TokenAnalyses(
                    surface=word,
                    analyses=(best,),
                ))
            else:
                # OOV fallback: try other backends
                results.append(self.analyze(word))
        return results

    def pipe(
        self,
        words: Iterator[str] | list[str],
        batch_size: int = 64,
        *,
        decompose_lvc: bool = False,
        handle_special_tokens: bool = False,
        handle_code_switch: bool = False,
    ) -> Iterator[TokenAnalyses]:
        """Batch-process words through the analyzer.

        Args:
            words: Iterable of Turkish words.
            batch_size: Not currently used for batching (reserved for future
                neural backend). Words are processed one at a time.
            decompose_lvc: Forward to :meth:`analyze`.
            handle_special_tokens: Forward to :meth:`analyze`.
            handle_code_switch: Forward to :meth:`analyze`.

        Yields:
            TokenAnalyses for each input word.
        """
        for word in words:
            yield self.analyze(
                word,
                decompose_lvc=decompose_lvc,
                handle_special_tokens=handle_special_tokens,
                handle_code_switch=handle_code_switch,
            )

    @property
    def cache(self) -> AnalysisCache:
        """Access the underlying cache for statistics."""
        return self._cache

    @property
    def cache_stats(self) -> dict[str, int | float]:
        """Return a snapshot of cache statistics.

        Keys: ``hits``, ``misses``, ``hit_rate``, ``memory_entries``,
        ``disk_entries``.
        """
        return self._cache.stats

    def enable_cache(
        self,
        memory_size: int = 100_000,
        disk_path: str | None = None,
    ) -> None:
        """Replace the current cache with a new (optionally disk-backed) one.

        Args:
            memory_size: Maximum entries in the memory LRU tier.
            disk_path: Path for a persistent :mod:`diskcache` SQLite store.
                ``None`` means memory-only.
        """
        self._cache = AnalysisCache(
            capacity=memory_size, disk_path=disk_path,
        )

    def close(self) -> None:
        """Release resources held by all backends.

        Terminates any long-lived subprocesses (e.g., TRMorph flookup).
        Call this when the analyzer is no longer needed, or use the
        context manager protocol instead.
        """
        for backend in self._backends:
            backend.close()

    def __enter__(self) -> MorphoAnalyzer:
        """Enter context manager — returns self."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager — releases all backend resources."""
        self.close()
