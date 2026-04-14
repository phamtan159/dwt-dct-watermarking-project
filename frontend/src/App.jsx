import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileAudio, CheckCircle2, AlertCircle, Loader2, Sparkles, PencilLine, Languages } from 'lucide-react';

const API_URL = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [targetText, setTargetText] = useState('');
  const [transcription, setTranscription] = useState('');
  const [aiAnswer, setAiAnswer] = useState('');
  const [aiType, setAiType] = useState(''); // 'gemini' or 'ollama'
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [isDragActive, setIsDragActive] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);

  const handleFileChange = (selectedFile) => {
    if (selectedFile) {
      setFile(selectedFile);
      setAudioUrl(URL.createObjectURL(selectedFile));
      setTranscription('');
      setAiAnswer('');
      setAiType('');
      setError('');
    }
  };

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragActive(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragActive(false);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragActive(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('audio/')) {
      handleFileChange(droppedFile);
    } else {
      setError('Vui lòng chọn file âm thanh hợp lệ.');
    }
  }, []);

  const handleSubmit = async (selectedAi) => {
    if (!file || !targetText) {
      setError('Vui lòng nhập nội dung định nói và chọn file âm thanh.');
      return;
    }

    setIsLoading(true);
    setAiType(selectedAi);
    setError('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_text', targetText);
    formData.append('ai_type', selectedAi);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      if (response.data.status === 'success') {
        setTranscription(response.data.transcription);
        setAiAnswer(response.data.ai_answer);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Lỗi kết nối Server.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div id="root">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card" style={{ maxWidth: '850px' }}>
        <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          <Sparkles size={40} color="var(--primary)" />
          <h1 style={{ marginTop: '0.5rem' }}>AI Pronunciation Auditor</h1>
        </div>

        <div className="input-section">
          <label style={{ color: 'var(--primary)', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <PencilLine size={18} />
            Bạn định nói từ/câu gì?
          </label>
          <input
            className="styled-input"
            value={targetText}
            onChange={(e) => setTargetText(e.target.value)}
            placeholder="Ví dụ: She sells seashells by the seashore"
          />
        </div>

        <div
          className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
          onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
          onClick={() => document.getElementById('fileInput').click()}
          style={{ marginTop: '1.5rem' }}
        >
          <input id="fileInput" type="file" accept="audio/*" onChange={(e) => handleFileChange(e.target.files[0])} style={{ display: 'none' }} />
          <div style={{ textAlign: 'center' }}>
            {file ? <CheckCircle2 color="#10b981" size={40} /> : <Upload size={40} />}
            <p style={{ marginTop: '10px' }}>{file ? file.name : 'Thả file âm thanh bạn đã nói vào đây'}</p>
          </div>
        </div>

        {audioUrl && (
          <div className="audio-preview">
            <audio src={audioUrl} controls style={{ width: '100%', marginTop: '1rem' }} />
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginTop: '2rem' }}>
          <button
            className="primary-btn gemini-btn"
            onClick={() => handleSubmit('gemini')}
            disabled={isLoading || !file || !targetText}
          >
            {isLoading && aiType === 'gemini' ? <Loader2 className="spin" /> : <Sparkles size={18} />}
            Gemini (AI Agent)
          </button>

          <button
            className="primary-btn ollama-btn"
            onClick={() => handleSubmit('ollama')}
            disabled={isLoading || !file || !targetText}
          >
            {isLoading && aiType === 'ollama' ? <Loader2 className="spin" /> : <Languages size={18} />}
            Local AI (Ollama)
          </button>
        </div>

        <AnimatePresence>
          {error && <div className="error-banner">{error}</div>}

          {transcription && (
            <div className="result-container-split">
              <div className="result-half">
                <h3><Languages size={18} /> Bạn định nói:</h3>
                <div className="text-box intended">{targetText}</div>
              </div>
              <div className="result-half">
                <h3><CheckCircle2 size={18} /> AI nghe thấy:</h3>
                <div className="text-box actual">{transcription}</div>
              </div>
            </div>
          )}

          {aiAnswer && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className={`ai-final-result ${aiType}-result`}>
              <h3><Sparkles size={18} /> Nhận xét từ {aiType === 'gemini' ? 'Gemini' : 'Ollama'}:</h3>
              <div className="ai-text-display">
                {aiAnswer.split('\n').map((line, i) => <p key={i}>{line}</p>)}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      <style dangerouslySetInnerHTML={{
        __html: `
        .styled-input {
          width: 100%;
          padding: 15px;
          background: rgba(255,255,255,0.05);
          border: 2px solid rgba(255,255,255,0.1);
          border-radius: 10px;
          color: white;
          font-size: 1.1rem;
          margin-top: 10px;
        }
        .styled-input:focus { border-color: var(--primary); outline: none; }
        .result-container-split {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
          margin-top: 2rem;
        }
        .text-box {
          padding: 15px;
          border-radius: 8px;
          min-height: 60px;
          font-weight: bold;
        }
        .intended { background: rgba(59, 130, 246, 0.1); color: #60a5fa; border: 1px solid #3b82f6; }
        .actual { background: rgba(16, 185, 129, 0.1); color: #34d399; border: 1px solid #10b981; }
        .gemini-btn { background: linear-gradient(135deg, #10b981 0%, #059669 100%); box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3); }
        .ollama-btn { background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3); }
        .ai-final-result {
          margin-top: 2rem;
          border-radius: 12px;
          padding: 20px;
        }
        .gemini-result { background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; }
        .ollama-result { background: rgba(99, 102, 241, 0.1); border: 1px solid #6366f1; }
        .ai-text-display { line-height: 1.7; color: #e2e8f0; }
        .error-banner { background: #ef444422; color: #ef4444; padding: 10px; border-radius: 8px; margin-top: 10px; text-align: center; }
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { from {transform: rotate(0deg)} to {transform: rotate(360deg)} }
      `}} />
    </div>
  );
}

export default App;
