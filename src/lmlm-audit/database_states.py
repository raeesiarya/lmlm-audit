import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


LOOKUP_PATTERNS = [
    r"\[dblookup\('((?:[^'\\]|\\.)+)',\s*'((?:[^'\\]|\\.)+)'\)\s*->",
    r"\[dblookup\('(.+?)',\s*'(.+?)'\)\s*->",
    r"<\|db_entity\|>(.+?)<\|db_relationship\|>(.+?)<\|db_return\|>",
]


class DatabaseState(str, Enum):
    FULL = "FULL"
    DEL_ON = "DEL-ON"
    DEL_OFF = "DEL-OFF"


@dataclass(frozen=True)
class TargetFact:
    fact_id: int | None
    subject: str
    relation: str
    object: str


def retrieval_enabled(state: DatabaseState) -> bool:
    return state is not DatabaseState.DEL_OFF


def target_fact_from_prompt_row(prompt_row: dict[str, Any]) -> TargetFact:
    return TargetFact(
        fact_id=prompt_row.get("fact_id"),
        subject=prompt_row["subject"],
        relation=prompt_row["relation"],
        object=prompt_row["gold_object"],
    )


def _normalize_field(text: str) -> str:
    return text.strip().casefold()


def is_deleted_triplet(triplet: tuple[str, str, str], target_fact: TargetFact) -> bool:
    subject, relation, obj = triplet
    return (
        _normalize_field(subject) == _normalize_field(target_fact.subject)
        and _normalize_field(relation) == _normalize_field(target_fact.relation)
        and _normalize_field(obj) == _normalize_field(target_fact.object)
    )


def extract_lookup_query(prompt: str) -> tuple[str, str]:
    matches = {
        tuple(match)
        for pattern in LOOKUP_PATTERNS
        for match in re.findall(pattern, prompt)
    }

    if not matches:
        raise ValueError(f"No valid dblookup pattern found in prompt: {prompt}")

    if len(matches) > 1:
        raise ValueError(f"Multiple dblookup matches found: {matches} in prompt: {prompt}")

    entity, relationship = matches.pop()
    return entity, relationship


def retrieve_triplet_candidates(
    topk_retriever: Any,
    entity: str,
    relation: str,
    threshold: float | None = None,
) -> list[tuple[str, str, str, float]]:
    query_text = (
        f"{topk_retriever._normalize_text(entity)} "
        f"{topk_retriever._normalize_text(relation)}"
    )
    query_embedding = topk_retriever.model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    distances, indices = topk_retriever.index.search(
        query_embedding,
        topk_retriever.top_k,
    )

    effective_threshold = (
        threshold if threshold is not None else topk_retriever.default_threshold
    )

    results: list[tuple[str, str, str, float]] = []
    for distance, index in zip(distances[0], indices[0]):
        if index == -1 or index not in topk_retriever.id_to_triplet:
            continue
        if distance < effective_threshold:
            continue

        subject, relation_name, value = topk_retriever.id_to_triplet[index]
        results.append((subject, relation_name, value, float(distance)))

    results.sort(key=lambda item: item[-1], reverse=True)
    return results


class AuditDatabaseManager:
    def __init__(
        self,
        base_db_manager: Any,
        state: DatabaseState,
        target_fact: TargetFact | None = None,
    ) -> None:
        self.base_db_manager = base_db_manager
        self.state = state
        self.target_fact = target_fact

        self.database_name = getattr(base_db_manager, "database_name", None)
        self.database_org_file = getattr(base_db_manager, "database_org_file", [])
        self.database = getattr(base_db_manager, "database", {})
        self.topk_retriever = getattr(base_db_manager, "topk_retriever", None)

    def init_topk_retriever(self, *args: Any, **kwargs: Any) -> None:
        if getattr(self.base_db_manager, "topk_retriever", None) is None:
            self.base_db_manager.init_topk_retriever(*args, **kwargs)
        self.topk_retriever = self.base_db_manager.topk_retriever

    def retrieve_from_database(self, prompt: str, threshold: float | None = None) -> str:
        if self.state is DatabaseState.FULL or self.target_fact is None:
            return self.base_db_manager.retrieve_from_database(prompt, threshold=threshold)

        entity, relationship = extract_lookup_query(prompt)
        self.init_topk_retriever()
        candidates = retrieve_triplet_candidates(
            self.topk_retriever,
            entity=entity,
            relation=relationship,
            threshold=threshold,
        )

        remaining_candidates = [
            candidate
            for candidate in candidates
            if not is_deleted_triplet(candidate[:3], self.target_fact)
        ]

        if not remaining_candidates:
            raise ValueError(
                f"No retrieval results for entity={entity!r}, relationship={relationship!r}"
            )

        return remaining_candidates[0][2]


def build_state_db_manager(
    base_db_manager: Any,
    prompt_row: dict[str, Any],
    state: DatabaseState,
) -> Any:
    if state is DatabaseState.FULL:
        return base_db_manager

    return AuditDatabaseManager(
        base_db_manager=base_db_manager,
        state=state,
        target_fact=target_fact_from_prompt_row(prompt_row),
    )
