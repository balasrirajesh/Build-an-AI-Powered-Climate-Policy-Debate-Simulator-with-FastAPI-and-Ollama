document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('start-btn');
    const topicInput = document.getElementById('topic');
    const roundsSelect = document.getElementById('rounds');
    const transcript = document.getElementById('transcript');
    const loading = document.getElementById('loading');
    const statusText = document.getElementById('status-text');

    const downloadBtn = document.getElementById('download-btn');
    let debateMessages = [];

    startBtn.addEventListener('click', async () => {
        const topic = topicInput.value.trim();
        const rounds = parseInt(roundsSelect.value);

        if (!topic) {
            alert('Please enter a debate topic.');
            return;
        }

        // Reset UI
        transcript.innerHTML = '';
        debateMessages = [];
        loading.style.display = 'block';
        startBtn.disabled = true;
        downloadBtn.style.display = 'none';
        statusText.innerText = 'Initializing debate agents...';

        try {
            const response = await fetch('/debate/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ topic, rounds })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to start debate. Is Ollama running?');
            }

            const data = await response.json();
            debateMessages = data.messages;
            loading.style.display = 'none';
            
            // Display messages with a slight delay for effect
            for (let i = 0; i < data.messages.length; i++) {
                const msg = data.messages[i];
                await displayMessage(msg, 400); // 400ms gap between each message reveal
            }

            // Show download button once finished
            downloadBtn.style.display = 'block';
            downloadBtn.style.animation = 'fadeIn 0.5s ease-out forwards';

        } catch (error) {
            console.error('Error:', error);
            loading.style.display = 'none';
            transcript.innerHTML = `
                <div class="message" style="border-left: 6px solid #ef4444; background: rgba(239, 68, 68, 0.1);">
                    <div class="message-header">
                        <div class="agent-name" style="color: #ef4444;">System Error</div>
                    </div>
                    <div class="message-content">
                        ${error.message}. Please ensure the Docker containers are running and Ollama has pulled the required model.
                    </div>
                </div>
            `;
            startBtn.disabled = false;
        } finally {
            startBtn.disabled = false;
        }
    });

    downloadBtn.addEventListener('click', () => {
        if (debateMessages.length === 0) return;

        const topic = topicInput.value.trim();
        let content = `# Climate Policy Debate Transcript\n`;
        content += `**Topic:** ${topic}\n`;
        content += `**Date:** ${new Date().toLocaleString()}\n\n`;
        content += `---\n\n`;

        debateMessages.forEach(msg => {
            content += `### Round ${msg.round} - ${msg.agent}\n`;
            content += `**Stance:** ${msg.stance.toUpperCase()}\n\n`;
            content += `${msg.message}\n\n`;
            content += `---\n\n`;
        });

        const blob = new Blob([content], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `debate_transcript_${topic.replace(/\s+/g, '_').toLowerCase()}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    async function displayMessage(msg, delay) {
        return new Promise(resolve => {
            setTimeout(() => {
                const msgEl = document.createElement('div');
                msgEl.className = `message ${msg.agent.toLowerCase()}`;
                
                const timestamp = new Date(msg.timestamp).toLocaleTimeString();
                
                msgEl.innerHTML = `
                    <div class="message-header">
                        <div class="agent-name">
                            <span class="agent-dot" style="width: 12px; height: 12px; border-radius: 50%; background: var(--${msg.agent.toLowerCase()}); box-shadow: 0 0 10px var(--${msg.agent.toLowerCase()})"></span>
                            ${msg.agent}
                        </div>
                        <div style="display: flex; gap: 0.5rem; align-items: center;">
                            <span class="stance ${msg.stance}">${msg.stance}</span>
                            <span class="round-badge">Round ${msg.round}</span>
                        </div>
                    </div>
                    <div class="message-content">
                        ${msg.message}
                    </div>
                    <div style="margin-top: 1rem; font-size: 0.75rem; color: var(--text-muted); text-align: right;">
                        ${timestamp}
                    </div>
                `;
                
                transcript.appendChild(msgEl);
                msgEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                resolve();
            }, delay);
        });
    }
});
