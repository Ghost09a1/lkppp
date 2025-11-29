"""
Basic tests for MyCandyLocal backend API
Run with: pytest tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core import create_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoints:
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "MyCandyLocal"
    
    def test_service_status(self, client):
        """Test service status endpoint"""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "backend" in data
        assert "llm" in data
        assert "tts" in data
        assert "image_gen" in data
        assert data["backend"] == "online"


class TestCharacterEndpoints:
    def test_list_characters(self, client):
        """Test listing characters"""
        response = client.get("/characters")
        assert response.status_code == 200
        data = response.json()
        assert "characters" in data
        assert isinstance(data["characters"], list)
    
    def test_create_character(self, client):
        """Test creating a new character"""
        char_data = {
            "name": "Test Character",
            "description": "A test character",
            "personality": "Friendly and helpful",
            "backstory": "Created for testing",
            "visual_style": "",
            "appearance_notes": "",
            "relationship_type": "friend",
            "dos": "",
            "donts": "",
            "voice_style": "",
            "voice_pitch_shift": 0.0,
            "voice_speed": 1.0,
            "voice_ref_path": "",
            "voice_youtube_url": "",
            "voice_model_path": "",
            "voice_training_status": "",
            "voice_error": "",
            "language": "en"
        }
        response = client.post("/characters", json=char_data)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert isinstance(data["id"], int)
        
        # Store for cleanup
        return data["id"]
    
    def test_get_character(self, client):
        """Test getting a specific character"""
        # First create a character
        char_id = self.test_create_character(client)
        
        # Then retrieve it
        response = client.get(f"/characters/{char_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == char_id
        assert data["name"] == "Test Character"
    
    def test_update_character(self, client):
        """Test updating a character"""
        # First create a character
        char_id = self.test_create_character(client)
        
        # Update it
        update_data = {
            "name": "Updated Test Character",
            "description": "Updated description",
            "personality": "Even more friendly",
            "backstory": "Updated backstory",
            "visual_style": "",
            "appearance_notes": "",
            "relationship_type": "friend",
            "dos": "",
            "donts": "",
            "voice_style": "",
            "voice_pitch_shift": 0.0,
            "voice_speed": 1.0,
            "voice_ref_path": "",
            "voice_youtube_url": "",
            "voice_model_path": "",
            "voice_training_status": "",
            "voice_error": "",
            "language": "en"
        }
        response = client.post(f"/characters/{char_id}/update", json=update_data)
        assert response.status_code == 200
        
        # Verify update
        response = client.get(f"/characters/{char_id}")
        data = response.json()
        assert data["name"] == "Updated Test Character"
        assert data["personality"] == "Even more friendly"


class TestChatEndpoints:
    def test_chat_without_llm(self, client):
        """Test chat endpoint (will fail gracefully without LLM)"""
        # First create a character
char_data = {
            "name": "Chat Test",
            "description": "Test",
            "personality": "Test",
            "backstory": "",
            "visual_style": "",
            "appearance_notes": "",
            "relationship_type": "",
            "dos": "",
            "donts": "",
            "voice_style": "",
            "voice_pitch_shift": 0.0,
            "voice_speed": 1.0,
            "voice_ref_path": "",
            "voice_youtube_url": "",
            "voice_model_path": "",
            "voice_training_status": "",
            "voice_error": "",
            "language": "en"
        }
        char_response = client.post("/characters", json=char_data)
        char_id = char_response.json()["id"]
        
        # Try to chat (will return fallback message without LLM)
        response = client.post(
            f"/chat/{char_id}",
            json={"message": "Hello"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        # Without LLM, should get fallback message
        assert "LLM unavailable" in data["reply"] or len(data["reply"]) > 0


class TestImageEndpoints:
    def test_generate_image_endpoint_exists(self, client):
        """Test that image generation endpoint exists"""
        # This will fail without SD.Next, but should return proper error
        response = client.post(
            "/generate_image",
            json={
                "prompt": "test",
                "negative": "",
                "steps": 20,
                "width": 512,
                "height": 768
            }
        )
        # Should return 200 even if image generation fails
        assert response.status_code in [200, 500]


class TestTTSEndpoints:
    def test_tts_endpoint_exists(self, client):
        """Test that TTS endpoint exists"""
        response = client.post(
            "/tts",
            json={
                "message": "Hello world",
                "character_id": None
            }
        )
        # Should return 200 even if TTS is offline
        assert response.status_code == 200
