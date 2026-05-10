"""
Tests for the AI Climate Policy Debate Simulator.
All heavy dependencies (chromadb, httpx/Ollama) are mocked so these run
without Docker or any local model installation.
"""
import sys
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# ---------------------------------------------------------------------------
# Patch heavy dependencies BEFORE any project code is imported
# ---------------------------------------------------------------------------
mock_chromadb = MagicMock()
mock_collection = MagicMock()
mock_collection.query.return_value = {"documents": [["Policy point A", "Policy point B"]]}
mock_chromadb.PersistentClient.return_value.create_collection.return_value = mock_collection
mock_chromadb.PersistentClient.return_value.delete_collection.return_value = None

sys.modules["chromadb"] = mock_chromadb
sys.modules["chromadb.utils"] = MagicMock()
sys.modules["chromadb.utils.embedding_functions"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()

# Now it is safe to import the app
from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
class TestHealthCheck:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_body(self):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Policy endpoints
# ---------------------------------------------------------------------------
class TestPolicyEndpoints:
    @pytest.mark.parametrize("country", ["usa", "eu", "china"])
    def test_get_valid_policy_returns_200(self, country):
        response = client.get(f"/policies/{country}")
        assert response.status_code == 200

    @pytest.mark.parametrize("country", ["usa", "eu", "china"])
    def test_get_valid_policy_body_has_country_field(self, country):
        response = client.get(f"/policies/{country}")
        assert "country" in response.json()

    @pytest.mark.parametrize("country", ["usa", "eu", "china"])
    def test_get_valid_policy_country_field_matches(self, country):
        response = client.get(f"/policies/{country}")
        assert response.json()["country"].lower() == country

    def test_get_invalid_policy_returns_404(self):
        response = client.get("/policies/invalid")
        assert response.status_code == 404

    def test_policy_has_key_positions(self):
        response = client.get("/policies/usa")
        body = response.json()
        assert "key_positions" in body
        assert isinstance(body["key_positions"], list)

    def test_policy_has_red_lines(self):
        response = client.get("/policies/usa")
        body = response.json()
        assert "red_lines" in body
        assert isinstance(body["red_lines"], list)


# ---------------------------------------------------------------------------
# Root endpoint (serves index.html)
# ---------------------------------------------------------------------------
class TestRootEndpoint:
    def test_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_html(self):
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# Debate endpoint – input validation
# ---------------------------------------------------------------------------
class TestDebateValidation:
    def test_rounds_above_max_rejected(self):
        response = client.post("/debate/start", json={"topic": "Test", "rounds": 6})
        assert response.status_code == 422

    def test_rounds_below_min_rejected(self):
        response = client.post("/debate/start", json={"topic": "Test", "rounds": 0})
        assert response.status_code == 422

    def test_missing_topic_rejected(self):
        response = client.post("/debate/start", json={"rounds": 2})
        assert response.status_code == 422

    def test_missing_rounds_rejected(self):
        response = client.post("/debate/start", json={"topic": "Test"})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Debate endpoint – full orchestration (Ollama mocked)
# ---------------------------------------------------------------------------
MOCK_LLM_RESPONSE = (
    "As the representative for this nation, we firmly stand behind our climate "
    "commitments and demand equal accountability from all parties.\nStance: supportive"
)


def make_mock_ollama_response(stance="supportive"):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "response": f"Our position is clear and consistent with our national policy.\nStance: {stance}"
    }
    return mock_resp


@pytest.fixture
def mock_ollama():
    """Fixture that patches httpx so no real Ollama call is made."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.post = AsyncMock(return_value=make_mock_ollama_response())
        mock_client_cls.return_value = mock_instance
        yield mock_instance


class TestDebateOrchestration:
    def test_one_round_returns_three_messages(self, mock_ollama):
        response = client.post("/debate/start", json={"topic": "Carbon Tax", "rounds": 1})
        assert response.status_code == 200
        assert len(response.json()["messages"]) == 3

    def test_two_rounds_returns_six_messages(self, mock_ollama):
        response = client.post("/debate/start", json={"topic": "Carbon Tax", "rounds": 2})
        assert response.status_code == 200
        assert len(response.json()["messages"]) == 6

    def test_three_rounds_returns_nine_messages(self, mock_ollama):
        response = client.post("/debate/start", json={"topic": "Carbon Tax", "rounds": 3})
        assert response.status_code == 200
        assert len(response.json()["messages"]) == 9

    def test_agent_turn_order_is_usa_eu_china(self, mock_ollama):
        response = client.post("/debate/start", json={"topic": "Carbon Tax", "rounds": 2})
        agents = [m["agent"] for m in response.json()["messages"]]
        assert agents == ["USA", "EU", "China", "USA", "EU", "China"]

    def test_round_numbers_are_correct(self, mock_ollama):
        response = client.post("/debate/start", json={"topic": "Carbon Tax", "rounds": 2})
        rounds = [m["round"] for m in response.json()["messages"]]
        assert rounds == [1, 1, 1, 2, 2, 2]

    def test_message_schema_has_all_required_keys(self, mock_ollama):
        response = client.post("/debate/start", json={"topic": "Carbon Tax", "rounds": 1})
        required_keys = {"round", "agent", "message", "stance", "timestamp"}
        for msg in response.json()["messages"]:
            assert required_keys.issubset(msg.keys())

    def test_stance_is_valid_value(self, mock_ollama):
        response = client.post("/debate/start", json={"topic": "Carbon Tax", "rounds": 1})
        valid_stances = {"supportive", "opposed", "neutral"}
        for msg in response.json()["messages"]:
            assert msg["stance"] in valid_stances

    def test_message_is_not_empty(self, mock_ollama):
        response = client.post("/debate/start", json={"topic": "Carbon Tax", "rounds": 1})
        for msg in response.json()["messages"]:
            assert msg["message"].strip() != ""

    def test_timestamp_is_iso8601_string(self, mock_ollama):
        from datetime import datetime
        response = client.post("/debate/start", json={"topic": "Carbon Tax", "rounds": 1})
        for msg in response.json()["messages"]:
            # Should not raise
            datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))

    def test_response_has_messages_key(self, mock_ollama):
        response = client.post("/debate/start", json={"topic": "Carbon Tax", "rounds": 1})
        assert "messages" in response.json()

    def test_messages_is_a_list(self, mock_ollama):
        response = client.post("/debate/start", json={"topic": "Carbon Tax", "rounds": 1})
        assert isinstance(response.json()["messages"], list)
