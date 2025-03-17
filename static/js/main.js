document.addEventListener('DOMContentLoaded', () => {
    // Tab Navigation
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            button.classList.add('active');
            document.getElementById(`${button.dataset.tab}-tab`).classList.add('active');
        });
    });

    // Chat Functionality
    const chatInput = document.getElementById('chat-input');
    const sendChat = document.getElementById('send-chat');
    const chatMessages = document.getElementById('chat-messages');

    sendChat.addEventListener('click', () => {
        const message = chatInput.value.trim();
        if (!message) return;

        appendMessage('user-message', message);
        chatInput.value = '';

        fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        })
        .then(response => response.json())
        .then(data => appendMessage('ai-message', data.response))
        .catch(error => console.error('Chat error:', error));
    });

    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChat.click();
    });

    function appendMessage(className, text) {
        const msg = document.createElement('div');
        msg.className = `chat-message ${className}`;
        msg.textContent = text;
        chatMessages.appendChild(msg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});