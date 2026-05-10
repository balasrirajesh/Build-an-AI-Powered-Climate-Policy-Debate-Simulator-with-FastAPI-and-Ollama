import httpx
import os
from core.rag_service import rag_service

class Debater:
    def __init__(self, country: str, model_name: str, base_url: str):
        self.country = country
        self.model_name = model_name
        self.base_url = base_url

    async def generate_response(self, topic: str, history: list) -> dict:
        """Generate a response based on the topic, history, and retrieved policy context."""
        # 1. Retrieve relevant policy context
        context_query = f"{topic} " + " ".join([m["message"] for m in history[-2:]])
        relevant_points = rag_service.query_policy(self.country.lower(), context_query)
        
        # 2. Construct prompt
        history_str = "\n".join([f"{m['agent']}: {m['message']}" for m in history])
        
        prompt = f"""You are the official debate representative for {self.country}.
You are debating the topic: {topic}.

Official Policy Context:
{chr(10).join(['- ' + p for p in relevant_points])}

Debate History:
{history_str}

Instructions:
1. Your response must be based on your country's official policy points provided above.
2. Maintain your country's persona and be firm but diplomatic.
3. Your response must be a single paragraph.
4. Conclude your response by explicitly stating your stance on the current state of the debate as 'supportive', 'opposed', or 'neutral'.

Response format:
[Your paragraph response here]
Stance: [supportive|opposed|neutral]
"""

        # 3. Call Ollama
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()["response"].strip()
                
                # Parse message and stance
                lines = result.split("\n")
                stance = "neutral"
                message_lines = []
                
                for line in lines:
                    if line.lower().startswith("stance:"):
                        stance_val = line.split(":", 1)[1].strip().lower().rstrip(".")
                        if stance_val in ["supportive", "opposed", "neutral"]:
                            stance = stance_val
                    elif line.strip():
                        message_lines.append(line)
                
                # If stance wasn't parsed correctly from a "Stance:" line, try to find it in the text
                if stance == "neutral" and message_lines:
                    last_line = message_lines[-1].lower()
                    if "supportive" in last_line:
                        stance = "supportive"
                    elif "opposed" in last_line:
                        stance = "opposed"
                
                # Remove the stance line from message if it exists
                clean_message_lines = [l for l in message_lines if not l.lower().startswith("stance:")]
                message = " ".join(clean_message_lines)

                return {
                    "agent": self.country,
                    "message": message,
                    "stance": stance
                }
        except Exception as e:
            print(f"Error in Debater ({self.country}): {str(e)}")
            return {
                "agent": self.country,
                "message": f"Error generating response: {str(e)}",
                "stance": "neutral"
            }
