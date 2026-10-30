import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  sendChatMessage,
  streamChatMessage,
  resetConversation,
  fetchConversations,
  fetchDocuments,
  uploadDocument,
  fetchDocumentStatus,
  deleteDocumentApi,
  updateUserProfile,
} from '../api.js';

describe('MediAssist Frontend API Client', () => {
  const originalFetch = global.fetch;
  const originalDocument = global.document;

  beforeEach(() => {
    vi.restoreAllMocks();
    global.document = {
      querySelector: vi.fn((selector) => {
        if (selector === 'meta[name="csrf-token"]') {
          return {
            getAttribute: (attr) => (attr === 'content' ? 'mock-csrf-token-12345' : null),
          };
        }
        return null;
      }),
    };
  });

  afterEach(() => {
    global.fetch = originalFetch;
    global.document = originalDocument;
  });

  describe('sendChatMessage', () => {
    it('sends message with CSRF token and parses structured response', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({
          response: 'Elevated blood pressure requires clinical evaluation.',
          sources: [{ source: 'Harrison Clinical Core', page_number: 42 }],
          triage: { urgency: 'routine' },
          conversation_id: 'conv-uuid-111',
          created_at: '2026-10-26T12:00:00Z',
        }),
      });

      const res = await sendChatMessage('My blood pressure is 140/90', 'conv-uuid-111');

      expect(global.fetch).toHaveBeenCalledWith('/chat', expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'X-CSRFToken': 'mock-csrf-token-12345',
        }),
        body: JSON.stringify({
          message: 'My blood pressure is 140/90',
          conversation_id: 'conv-uuid-111',
        }),
      }));

      expect(res.text).toBe('Elevated blood pressure requires clinical evaluation.');
      expect(res.sources).toHaveLength(1);
      expect(res.conversation_id).toBe('conv-uuid-111');
    });

    it('throws descriptive error on server failure', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({ error: 'Message cannot be empty.' }),
      });

      await expect(sendChatMessage('   ')).rejects.toThrow('Message cannot be empty.');
    });
  });

  describe('Document Attachments API', () => {
    it('fetchDocuments returns array of attached documents', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          documents: [
            { id: 'doc-1', filename: 'Echo_Report.pdf', status: 'ready', page_count: 2 },
          ],
        }),
      });

      const docs = await fetchDocuments('conv-123');
      expect(docs).toHaveLength(1);
      expect(docs[0].filename).toBe('Echo_Report.pdf');
      expect(docs[0].status).toBe('ready');
    });

    it('uploadDocument posts FormData with ?async=1 query flag', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 201,
        json: async () => ({
          status: 'processing',
          document: { id: 'doc-async-1', status: 'processing', filename: 'Lab_Report.pdf' },
        }),
      });

      const fakeFile = new File(['dummy content'], 'Lab_Report.pdf', { type: 'application/pdf' });
      const res = await uploadDocument('conv-456', fakeFile, true);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/conversations/conv-456/upload?async=1',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({ 'X-CSRFToken': 'mock-csrf-token-12345' }),
        })
      );
      expect(res.status).toBe('processing');
      expect(res.document.id).toBe('doc-async-1');
    });

    it('fetchDocumentStatus queries endpoint and returns status record', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          status: 'ready',
          document: { id: 'doc-999', status: 'ready', chunk_count: 3 },
        }),
      });

      const status = await fetchDocumentStatus('conv-456', 'doc-999');
      expect(global.fetch).toHaveBeenCalledWith('/api/conversations/conv-456/documents/doc-999/status');
      expect(status.status).toBe('ready');
      expect(status.document.chunk_count).toBe(3);
    });

    it('deleteDocumentApi issues DELETE with CSRF token', async () => {
      global.fetch = vi.fn().mockResolvedValue({ ok: true, status: 200 });

      const ok = await deleteDocumentApi('conv-456', 'doc-999');
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/conversations/conv-456/documents/doc-999',
        expect.objectContaining({
          method: 'DELETE',
          headers: expect.objectContaining({ 'X-CSRFToken': 'mock-csrf-token-12345' }),
        })
      );
      expect(ok).toBe(true);
    });
  });

  describe('User Profile & Explanation Perspective', () => {
    it('updateUserProfile sends PATCH request with academic year and target exam', async () => {
      global.fetch = vi.fn().mockResolvedValue({ ok: true, status: 200 });

      const ok = await updateUserProfile('2nd_year', 'neet_pg_next');
      expect(global.fetch).toHaveBeenCalledWith('/api/profile', expect.objectContaining({
        method: 'PATCH',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'X-CSRFToken': 'mock-csrf-token-12345',
        }),
        body: JSON.stringify({
          academic_year: '2nd_year',
          target_exam: 'neet_pg_next',
        }),
      }));
      expect(ok).toBe(true);
    });
  });

  describe('streamChatMessage SSE parser', () => {
    it('yields incremental SSE events from readable stream', async () => {
      const ssePayload = [
        'event: start\ndata: {"conversation_id": "conv-stream-1"}\n\n',
        'event: chunk\ndata: {"chunk": "Blood pressure "}\n\n',
        'event: chunk\ndata: {"chunk": "is normal."}\n\n',
        'event: complete\ndata: {"full_text": "Blood pressure is normal."}\n\n',
      ].join('');

      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(ssePayload));
          controller.close();
        },
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: stream,
      });

      const events = [];
      for await (const evt of streamChatMessage('Check BP', 'conv-stream-1')) {
        events.push(evt);
      }

      expect(events).toHaveLength(4);
      expect(events[0].event).toBe('start');
      expect(events[1].data.chunk).toBe('Blood pressure ');
      expect(events[2].data.chunk).toBe('is normal.');
      expect(events[3].event).toBe('complete');
    });
  });
});
