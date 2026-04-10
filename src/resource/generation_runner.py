"""Generation 0 bootstrap runner for TR-Gold-Morph resource database."""
from __future__ import annotations
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class GenerationRunner:
    """Orchestrates progressive resource generation.

    Generation 0 (bootstrap):
      1. Clone IMST and UniMorph if not present
      2. Import BOUN treebank (already present)
      3. Import IMST treebank
      4. Import UniMorph Turkish
      5. Run Zeyrek on all unique surfaces
      6. Re-evaluate tiers by cross-source agreement
      7. Print stats summary

    Args:
        project_root: Root directory of the project. Defaults to the
            directory three levels above this file.
        db_path: Path to SQLite database. Defaults to
            ``data/resource/tr_gold_morph.db``.
    """

    def __init__(
        self,
        project_root: Path | str | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.root = Path(project_root)
        self.db_path = Path(db_path) if db_path else self.root / "data/resource/tr_gold_morph.db"

    def run(self, generation: int = 0) -> None:
        """Run the specified generation.

        Args:
            generation: Generation number (0 = bootstrap, 1 = Wikipedia,
                2 = OSCAR).
        """
        if generation == 0:
            self.run_generation_0()
        elif generation == 1:
            self.run_generation_1()
        elif generation == 2:
            self.run_generation_2()
        else:
            raise NotImplementedError(f"Generation {generation} not yet implemented")

    def run_generation_0(self) -> None:  # noqa: PLR0912
        """Execute Generation 0 bootstrap."""
        from resource.importers.boun import import_boun
        from resource.importers.imst import import_imst, IMST_CLONE_URL, IMST_DEFAULT_DIR
        from resource.importers.unimorph import import_unimorph, UNIMORPH_CLONE_URL, UNIMORPH_DEFAULT_DIR
        from resource.importers.zeyrek_bulk import analyze_bulk
        from resource.quality_check import tier_from_entries
        from resource.schema import MorphDatabase, MorphEntry
        from resource.normalizer import normalize_surface

        print("=== TR-Gold-Morph Generation 0 Bootstrap ===\n")

        # 1. Ensure external data directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 2. Clone IMST if not present
        imst_dir = self.root / IMST_DEFAULT_DIR
        if not imst_dir.exists():
            print(f"Cloning IMST treebank → {imst_dir} ...")
            subprocess.run(
                ["git", "clone", "--depth", "1", IMST_CLONE_URL, str(imst_dir)],
                check=True,
                capture_output=True,
            )
        else:
            print(f"IMST treebank already present at {imst_dir}")

        # 3. Clone UniMorph if not present
        unimorph_dir = self.root / UNIMORPH_DEFAULT_DIR
        if not unimorph_dir.exists():
            print(f"Cloning UniMorph Turkish → {unimorph_dir} ...")
            subprocess.run(
                ["git", "clone", "--depth", "1", UNIMORPH_CLONE_URL, str(unimorph_dir)],
                check=True,
                capture_output=True,
            )
        else:
            print(f"UniMorph already present at {unimorph_dir}")

        with MorphDatabase(self.db_path) as db:
            # 4. Import BOUN
            boun_dir = self.root / "data/external/boun_treebank"
            print(f"\n[1/4] Importing BOUN treebank from {boun_dir} ...")
            boun_count = import_boun(boun_dir, db)
            print(f"      BOUN: {boun_count:,} entries parsed")

            # 5. Import IMST
            print(f"\n[2/4] Importing IMST treebank from {imst_dir} ...")
            imst_count = import_imst(imst_dir, db)
            print(f"      IMST: {imst_count:,} entries parsed")

            # 6. Import UniMorph
            print(f"\n[3/4] Importing UniMorph Turkish from {unimorph_dir} ...")
            uni_count = import_unimorph(unimorph_dir, db)
            print(f"      UniMorph: {uni_count:,} entries parsed")

            # 7. Run Zeyrek on all unique surfaces
            all_surfaces = db.get_all_surfaces()
            print(f"\n[4/4] Running Zeyrek on {len(all_surfaces):,} unique surfaces ...")
            zeyrek_results = analyze_bulk(all_surfaces)

            zeyrek_entries: list[MorphEntry] = []
            for surface, canonical in zeyrek_results.items():
                if canonical is None:
                    continue
                parts = canonical.split()
                lemma = parts[0] if parts else surface
                pos = "NOUN"  # default; Zeyrek POS comes from canonical tags
                for part in parts[1:]:
                    if part in ("+Noun", "+Verb", "+Adj", "+Adv"):
                        pos = part.lstrip("+").upper()
                        break
                zeyrek_entries.append(MorphEntry(
                    surface=surface,
                    lemma=lemma,
                    canonical_tags=canonical,
                    pos=pos,
                    source="zeyrek",
                    confidence=0.9,
                    frequency=1,
                    tier="bronze",
                ))

            db.bulk_insert(zeyrek_entries)
            print(f"      Zeyrek: {len(zeyrek_entries):,} entries inserted")

            # 8. Re-evaluate tiers by cross-source agreement
            print("\nRe-evaluating tiers by cross-source agreement ...")
            all_surfaces_updated = db.get_all_surfaces()
            updated = 0
            for surface in all_surfaces_updated:
                entries = db.query_surface(surface)
                if len(entries) <= 1:
                    continue  # nothing to compare
                tier, agreement = tier_from_entries(entries)
                for entry in entries:
                    db.update_tier(surface, entry.source, tier, agreement)
                updated += 1
            print(f"      Updated tiers for {updated:,} multi-source surfaces")

            # 9. Print summary
            stats = db.get_stats()
            total = stats["total"]
            print(f"\n{'='*50}")
            print(f"Generation 0 complete!")
            print(f"  Total entries    : {total:,}")
            print(f"  Unique surfaces  : {stats['unique_surfaces']:,}")
            print(f"  Unique lemmas    : {stats['unique_lemmas']:,}")
            print(f"\n  By tier:")
            for tier, count in sorted(stats["by_tier"].items()):
                print(f"    {tier:8s}: {count:,}")
            print(f"\n  By source:")
            for source, count in sorted(stats["by_source"].items()):
                print(f"    {source:10s}: {count:,}")
            print(f"{'='*50}")

            if total < 200_000:
                print(f"\nWARNING: Total {total:,} < 200,000 target. "
                      "Check that IMST and UniMorph clones succeeded.")


    def run_generation_1(self) -> None:
        """Execute Generation 1 — Turkish Wikipedia expansion.

        Pre-condition: Generation 0 has been run (DB exists with BOUN/IMST/UniMorph
        entries). Streams Turkish Wikipedia through Zeyrek and inserts new surfaces
        not already in the DB.

        Target: ≥500K total entries after this generation.
        """
        from resource.importers.wikipedia import import_wikipedia
        from resource.schema import MorphDatabase

        print("=== TR-Gold-Morph Generation 1 — Wikipedia Expansion ===\n")

        if not self.db_path.exists():
            print(
                f"ERROR: DB not found at {self.db_path}. "
                "Run Generation 0 first: --generation 0"
            )
            raise FileNotFoundError(f"DB not found: {self.db_path}")

        with MorphDatabase(self.db_path) as db:
            stats_before = db.get_stats()
            print(f"  DB before Gen 1: {stats_before['total']:,} entries, "
                  f"{stats_before['unique_surfaces']:,} unique surfaces")
            print()

            new_count = import_wikipedia(db)

            stats_after = db.get_stats()
            total = stats_after["total"]
            print(f"\n{'='*50}")
            print("Generation 1 complete!")
            print(f"  New entries inserted : {new_count:,}")
            print(f"  Total entries        : {total:,}")
            print(f"  Unique surfaces      : {stats_after['unique_surfaces']:,}")
            print(f"\n  By tier:")
            for tier, count in sorted(stats_after["by_tier"].items()):
                print(f"    {tier:8s}: {count:,}")
            print(f"\n  By source:")
            for source, count in sorted(stats_after["by_source"].items()):
                print(f"    {source:10s}: {count:,}")
            print(f"{'='*50}")

            if total < 500_000:
                print(
                    f"\nWARNING: Total {total:,} < 500,000 target. "
                    "Wikipedia may have been limited (max_articles set?) or Zeyrek "
                    "had low coverage."
                )


    def run_generation_2(self) -> None:
        """Execute Generation 2 — OSCAR Turkish corpus expansion.

        Pre-condition: Generation 1 has been run (DB has ≥500K entries from
        Wikipedia). Streams OSCAR-2301 Turkish through Zeyrek and inserts new
        surfaces not already in the DB.

        Target: ≥2M total entries after this generation.
        """
        from resource.importers.oscar import import_oscar
        from resource.schema import MorphDatabase

        print("=== TR-Gold-Morph Generation 2 — OSCAR Expansion ===\n")

        if not self.db_path.exists():
            print(
                f"ERROR: DB not found at {self.db_path}. "
                "Run Generation 1 first: --generation 1"
            )
            raise FileNotFoundError(f"DB not found: {self.db_path}")

        with MorphDatabase(self.db_path) as db:
            stats_before = db.get_stats()
            print(f"  DB before Gen 2: {stats_before['total']:,} entries, "
                  f"{stats_before['unique_surfaces']:,} unique surfaces")
            print()

            new_count = import_oscar(db)

            stats_after = db.get_stats()
            total = stats_after["total"]
            print(f"\n{'='*50}")
            print("Generation 2 complete!")
            print(f"  New entries inserted : {new_count:,}")
            print(f"  Total entries        : {total:,}")
            print(f"  Unique surfaces      : {stats_after['unique_surfaces']:,}")
            print(f"\n  By tier:")
            for tier, count in sorted(stats_after["by_tier"].items()):
                print(f"    {tier:8s}: {count:,}")
            print(f"\n  By source:")
            for source, count in sorted(stats_after["by_source"].items()):
                print(f"    {source:10s}: {count:,}")
            print(f"{'='*50}")

            if total < 2_000_000:
                print(
                    f"\nWARNING: Total {total:,} < 2,000,000 target. "
                    "OSCAR may have been limited (max_docs set?) or Zeyrek "
                    "had low coverage."
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run resource generation")
    parser.add_argument("--generation", type=int, default=0, help="Generation number (default: 0)")
    parser.add_argument("--db", default=None, help="Custom DB path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runner = GenerationRunner(db_path=args.db)
    runner.run(generation=args.generation)
