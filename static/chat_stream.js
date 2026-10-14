(function (root) {
    'use strict';
    async function readChatResponse(response, onSafety) {
        if (!response.ok) throw new Error('chat_http_' + response.status);
        if (!(response.headers.get('Content-Type') || '').includes('application/x-ndjson')) return response.json();
        const reader = response.body.getReader(), decoder = new TextDecoder();
        let buffer = '', final = null;
        function line(text) {
            if (!text.trim()) return;
            const event = JSON.parse(text);
            if (event.kind === 'safety_notice') onSafety(event.response);
            else if (event.kind === 'final') final = event;
            else if (event.kind === 'error') throw new Error(event.response);
            else throw new Error('unknown_chat_event');
        }
        try {
            while (true) {
                const { value, done } = await reader.read();
                buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
                let end;
                while ((end = buffer.indexOf('\n')) >= 0) { line(buffer.slice(0, end)); buffer = buffer.slice(end + 1); }
                if (done) break;
            }
            line(buffer);
            if (!final) throw new Error('chat_stream_incomplete');
            return final;
        } catch (error) {
            await reader.cancel().catch(() => {});
            throw error;
        } finally { reader.releaseLock(); }
    }
    if (typeof module !== 'undefined' && module.exports) module.exports = { readChatResponse };
    else root.readChatResponse = readChatResponse;
})(typeof window !== 'undefined' ? window : globalThis);
