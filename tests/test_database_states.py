import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/lmlm-audit"))

from database_states import (
    AuditDatabaseManager,
    DatabaseState,
    TargetFact,
    extract_lookup_query,
    is_deleted_triplet,
    retrieval_enabled,
)


class FakeModel:
    def encode(self, *_args, **_kwargs):
        return [[1.0]]


class FakeIndex:
    def __init__(self, indices: list[int], distances: list[float]) -> None:
        self.indices = indices
        self.distances = distances

    def search(self, _query_embedding, _top_k):
        return [self.distances], [self.indices]


class FakeRetriever:
    def __init__(self) -> None:
        self.top_k = 3
        self.default_threshold = 0.6
        self.model = FakeModel()
        self.index = FakeIndex(
            indices=[0, 1, 2],
            distances=[0.95, 0.90, 0.80],
        )
        self.id_to_triplet = {
            0: ("Hexol", "First Described By", "Jørgensen"),
            1: ("Hexol", "Structure Recognized By", "Werner"),
            2: ("Jocelyne Girard-Bujold", "Term End", "2004"),
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.lower().strip()


class FakeBaseManager:
    def __init__(self) -> None:
        self.database_name = "fake"
        self.database_org_file = []
        self.database = {}
        self.topk_retriever = FakeRetriever()

    def init_topk_retriever(self, *args, **kwargs) -> None:
        return None

    def retrieve_from_database(self, _prompt: str, threshold=None) -> str:
        return "Jørgensen"


def test_retrieval_enabled() -> None:
    assert retrieval_enabled(DatabaseState.FULL) is True
    assert retrieval_enabled(DatabaseState.DEL_ON) is True
    assert retrieval_enabled(DatabaseState.DEL_OFF) is False


def test_extract_lookup_query() -> None:
    entity, relation = extract_lookup_query(
        "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
    )
    assert entity == "Hexol"
    assert relation == "First Described By"


def test_is_deleted_triplet() -> None:
    target_fact = TargetFact(
        fact_id=10,
        subject="Hexol",
        relation="First Described By",
        object="Jørgensen",
    )
    assert (
        is_deleted_triplet(
            ("Hexol", "First Described By", "Jørgensen"),
            target_fact,
        )
        is True
    )
    assert (
        is_deleted_triplet(
            ("Hexol", "Structure Recognized By", "Werner"),
            target_fact,
        )
        is False
    )


def test_audit_database_manager_filters_deleted_fact() -> None:
    base_manager = FakeBaseManager()
    target_fact = TargetFact(
        fact_id=10,
        subject="Hexol",
        relation="First Described By",
        object="Jørgensen",
    )
    audit_manager = AuditDatabaseManager(
        base_db_manager=base_manager,
        state=DatabaseState.DEL_ON,
        target_fact=target_fact,
    )

    value = audit_manager.retrieve_from_database(
        "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
    )
    assert value == "Werner"
