import React, { useRef, useState, useEffect, type RefObject, useCallback } from 'react';
import './App.css';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence } from 'framer-motion';

// --- Helper Components ---
const Spinner = () => (
  <div className="flex justify-center items-center h-full">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
  </div>
);

type RenderMode = 'preview' | 'text';

// --- Props Type Definitions ---

type InputSectionProps = {
  input: string;
  setInput: (value: string) => void;
  handleSend: () => void;
  handleCancel: () => void;
  isLoading: boolean;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
};

type LongTextViewerProps = {
  renderMode: RenderMode;
  isHtmlContent: boolean;
  longTextContent: string;
  iframeRef: RefObject<HTMLIFrameElement | null>;
  longTextViewerRef: RefObject<HTMLDivElement | null>;
  isEditing: boolean;
  editedContent: string;
  onContentChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
};

// --- Child Components ---

const InputSection: React.FC<InputSectionProps> = ({ input, setInput, handleSend, handleCancel, isLoading, textareaRef }) => (
  <>
    <textarea
      ref={textareaRef}
      value={input}
      onChange={(e) => setInput(e.target.value)}
      disabled={isLoading}
      placeholder="카에데에게 말걸어주세요!"
      className="w-full p-4 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-transparent transition resize-none min-h-[120px] overflow-hidden disabled:bg-gray-100 disabled:cursor-not-allowed"
    />
    <div className="mt-4 flex gap-3 justify-end">
      <button
        onClick={handleSend}
        disabled={isLoading || input.trim() === ''}
        className="bg-indigo-500 hover:bg-indigo-600 text-white font-semibold py-2 px-6 rounded-xl transition disabled:opacity-50 disabled:cursor-not-allowed"
      >
        전송
      </button>
      {isLoading && (
        <button
          onClick={handleCancel}
          className="bg-red-400 hover:bg-red-500 text-white font-semibold py-2 px-4 rounded-xl transition"
        >
          중단
        </button>
      )}
    </div>
  </>
);

const HtmlPreview = React.forwardRef<HTMLIFrameElement, { htmlContent: string }>(
  ({ htmlContent }, ref) => (
    <iframe
      ref={ref}
      srcDoc={htmlContent}
      style={{ width: '100%', height: '100%', border: 'none', display: 'block', borderRadius: '0.75rem' }}
      title="HTML Preview"
      sandbox="allow-scripts allow-same-origin"
    />
  )
);

const LongTextViewer: React.FC<LongTextViewerProps> = ({ renderMode, isHtmlContent, longTextContent, iframeRef, longTextViewerRef, isEditing, editedContent, onContentChange }) => {
  return (
    <div ref={longTextViewerRef} className="h-full">
      {renderMode === 'preview' ? (
        isHtmlContent ? (
          <HtmlPreview htmlContent={longTextContent} ref={iframeRef} />
        ) : (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{longTextContent}</ReactMarkdown>
        )
      ) : isEditing ? (
        <textarea
          value={editedContent}
          onChange={onContentChange}
          className="w-full h-full p-3 border border-indigo-300 rounded-lg bg-white text-gray-900 font-mono text-sm resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
      ) : (
        <pre className="whitespace-pre-wrap break-words">{longTextContent}</pre>
      )}
    </div>
  );
};


// --- Main App Component ---
function App() {
  const FILE_NOT_FOUND_MESSAGE = '`longText.txt` 파일을 찾을 수 없습니다. 백엔드에서 파일을 생성했는지 확인해주세요.';
  const [backendStatus, setBackendStatus] = useState<'connected' | 'error' | 'connecting'>('connecting');
  const [input, setInput] = useState('');
  const [finalResponse, setFinalResponse] = useState('');
  const [streamingThoughts, setStreamingThoughts] = useState('');
  const [savedThoughts, setSavedThoughts] = useState('');
  const [longTextContent, setLongTextContent] = useState('');
  const [renderMode, setRenderMode] = useState<RenderMode>('preview');
  const [isHtmlContent, setIsHtmlContent] = useState(false);
  const [isLongTextPaneOpen, setIsLongTextPaneOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [elapsedTime, setElapsedTime] = useState<number | null>(null);
  const [isThinking, setIsThinking] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [isPollingPaused, setIsPollingPaused] = useState(false);
  const [showThoughtsToggle, setShowThoughtsToggle] = useState(false);
  const [isThoughtsOpen, setIsThoughtsOpen] = useState(false);
  
  // ✨ 1. 세션 ID를 저장할 상태를 추가합니다.
  const [sessionId, setSessionId] = useState<string | null>(null);

  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const responseContainerRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const longTextContentRef = useRef<HTMLDivElement>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const longTextViewerRef = useRef<HTMLDivElement>(null);
  const scrollToBottomButtonRef = useRef<HTMLButtonElement>(null);

  const adjustHeight = useCallback(() => {
    const container = longTextContentRef.current;
    if (!container) return;

    const computedStyle = window.getComputedStyle(container);
    const verticalPadding = parseFloat(computedStyle.paddingTop) + parseFloat(computedStyle.paddingBottom);
    let contentHeight = 0;
    const iframe = iframeRef.current;
    const viewer = longTextViewerRef.current;

    if (renderMode === 'preview' && isHtmlContent && iframe?.contentWindow) {
      contentHeight = iframe.contentWindow.document.documentElement.scrollHeight;
    } else if (viewer) {
      contentHeight = viewer.scrollHeight;
    }

    if (contentHeight > 0) {
      container.style.minHeight = `${contentHeight + verticalPadding}px`;
    }
  }, [renderMode, isHtmlContent]);

  useEffect(() => {
    const fetchAndSetText = async () => {
      if (backendStatus === 'error' && !isPollingPaused) {
        // 이미 에러 상태임을 인지하고 있다면 불필요한 로그 출력을 막기 위해 조용히 종료
        return;
      }
      try {
        // 캐시 문제를 피하기 위해 타임스탬프를 추가합니다.
        const response = await fetch(`/longText.txt?t=${Date.now()}`);

        if (response.ok) {
          if (backendStatus !== 'connected') {
            setBackendStatus('connected');
          }
          const text = await response.text();

          // Vite 개발 서버의 SPA Fallback 응답(index.html)인지 확인하는 가장 확실한 방법입니다.
          // 실제 longText.txt 파일에는 이 내용이 포함될 가능성이 거의 없습니다.
          // '/@react-refresh'는 Vite 개발 서버가 주입하는 스크립트로, SPA fallback을 식별하는 데 더 신뢰할 수 있습니다.
          if (text.includes('/@react-refresh')) {
            console.warn("Vite SPA fallback이 감지되었습니다. 응답을 무시합니다.");
            return; // Fallback 응답은 무시하고 다음 폴링을 기다립니다.
          }

          // 응답이 비어있을 경우, 파일을 찾을 수 없는 것과 동일하게 처리합니다.
          if (text.trim() === '') {
            setLongTextContent(currentText => {
              if (currentText === FILE_NOT_FOUND_MESSAGE) return currentText;
              setIsHtmlContent(false);
              setIsLongTextPaneOpen(true);
              return FILE_NOT_FOUND_MESSAGE;
            });
            return;
          }

          // 여기까지 왔다면 올바른 파일 내용을 받은 것입니다.
          setLongTextContent(currentText => {
            if (text === currentText) return currentText;

            const isHtml = text.trim().toLowerCase().startsWith('<!doctype html') || text.trim().toLowerCase().startsWith('<html');
            setIsHtmlContent(isHtml);
            // 새로운 내용이 도착하면(비어 있더라도) 패널을 엽니다.
            // 사용자가 수동으로 닫으면 다음 폴링에서 내용이 동일할 경우 다시 열리지 않습니다.
            setIsLongTextPaneOpen(true);
            return text;
          });
        } else if (response.status === 404) {
          // stale closure를 피하기 위해 함수형 업데이트를 사용합니다.
          setLongTextContent(currentText => {
            if (currentText === FILE_NOT_FOUND_MESSAGE) return currentText;
            setIsHtmlContent(false);
            setIsLongTextPaneOpen(true);
            return FILE_NOT_FOUND_MESSAGE;
          });
        }
      } catch (error) {
        // ✨ 4. fetch 자체에서 오류 발생 시 (ECONNRESET, ECONNREFUSED 등)
        console.error("백엔드 연결 오류:", error);
        
        // 연결 상태를 'error'로 설정
        if (backendStatus !== 'error') {
          setBackendStatus('error');
          setLongTextContent('⚠️ 백엔드 서버에 연결할 수 없습니다. 잠시 후 다시 시도합니다...');
          setIsHtmlContent(false);
          setIsLongTextPaneOpen(true);
        }
      }
    };

    // ✨ --- 핵심 수정: 폴링 로직 변경 ---
    let intervalId: number;

    if (backendStatus === 'error') {
      // 에러 상태일 경우, 5초 후에 상태를 'connecting'으로 바꿔서 재시도를 유발합니다.
      intervalId = window.setTimeout(() => {
        setBackendStatus('connecting');
      }, 5000);
    } else {
      // 정상 또는 연결 중 상태일 경우, 2초 간격으로 계속 폴링합니다.
      if (!isPollingPaused) {
        fetchAndSetText(); // 즉시 실행
        intervalId = window.setInterval(fetchAndSetText, 2000);
      }
    }
    // ------------------------------------

    return () => {
      if (intervalId) {
        if (backendStatus === 'error') {
          clearTimeout(intervalId);
        } else {
          clearInterval(intervalId);
        }
      }
    };
  }, [isPollingPaused, FILE_NOT_FOUND_MESSAGE, backendStatus]);

  useEffect(() => {
    const checkScrollButton = () => {
      setTimeout(() => {
        if (!scrollToBottomButtonRef.current) return;
        const { scrollTop, scrollHeight, clientHeight } = document.documentElement;
        const hasScrollbar = scrollHeight > clientHeight;
        const isAtBottom = scrollHeight - scrollTop - clientHeight < 1;
        scrollToBottomButtonRef.current.style.display = (hasScrollbar && !isAtBottom) ? 'flex' : 'none';
      }, 100);
    };

    window.addEventListener('scroll', checkScrollButton);
    checkScrollButton();

    return () => window.removeEventListener('scroll', checkScrollButton);
  }, [isLongTextPaneOpen, renderMode]);

  useEffect(() => {
    const container = longTextContentRef.current;
    if (!container) return;
    container.style.minHeight = '';
    if (!isLongTextPaneOpen) return;

    let observer: ResizeObserver | null = null;
    const iframe = iframeRef.current;
    const viewer = longTextViewerRef.current;

    if (renderMode !== 'preview' || !isHtmlContent) {
      if (viewer) {
        observer = new ResizeObserver(adjustHeight);
        observer.observe(viewer);
        adjustHeight();
      }
    } else if (iframe) {
      const setupIframeObserver = () => {
        const body = iframe?.contentWindow?.document.body;
        if (body) {
          observer = new ResizeObserver(adjustHeight);
          observer.observe(body);
          setTimeout(adjustHeight, 150);
        }
      };

      if (iframe.contentWindow?.document.readyState === 'complete') {
        setupIframeObserver();
      } else {
        iframe.addEventListener('load', setupIframeObserver);
      }
      return () => {
        if (iframe) iframe.removeEventListener('load', setupIframeObserver);
        if (observer) observer.disconnect();
      };
    }
    return () => {
      if (observer) observer.disconnect();
    };
  }, [isLongTextPaneOpen, longTextContent, renderMode, isHtmlContent, adjustHeight]);


  const handleSend = async () => {
    const currentInput = input.trim();
    if (!currentInput) return;

    setFinalResponse('');
    setStreamingThoughts('');
    setSavedThoughts('');
    setElapsedTime(null);
    setIsLoading(true);
    setIsThinking(true);
    setShowThoughtsToggle(false);
    setIsThoughtsOpen(false);

    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;
    const start = performance.now();

    try {
      // ✨ 2. API 요청 시 현재 sessionId를 함께 보냅니다.
      // sessionId가 null이면 백엔드에서 새로 생성할 것입니다.
      const response = await fetch('http://localhost:8000/gemini_stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            prompt: currentInput,
            session_id: sessionId 
        }),
        signal,
      });

      if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error("응답 본문을 읽을 수 없습니다.");

      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let firstThoughtReceived = false;
      let finalAnswerStarted = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const messages = buffer.split('\n\n');
        buffer = messages.pop() || '';

        for (const message of messages) {
          if (message.startsWith('data: ')) {
            const jsonString = message.substring(6);
            if (jsonString === '[DONE]') break;
            const data = JSON.parse(jsonString);

            // ✨ 3. 백엔드로부터 받은 session_id를 상태에 저장합니다.
            if (data.type === 'session_info' && data.session_id) {
                setSessionId(data.session_id);
            } else if (data.type === 'thought_stream' && typeof data.char === 'string') {
              if (!firstThoughtReceived) {
                setIsThinking(false);
                firstThoughtReceived = true;
              }
              if (!finalAnswerStarted) {
                setStreamingThoughts(prev => prev + data.char);
                setSavedThoughts(prev => prev + data.char);
              }
            } else if (data.type === 'thought_stream_end') {
              if (!finalAnswerStarted) {
                setStreamingThoughts(prev => prev + '\n\n---\n\n');
                setSavedThoughts(prev => prev + '\n\n---\n\n');
              }
            } else if (data.type === 'final_answer_stream' && typeof data.char === 'string') {
              if (!finalAnswerStarted) {
                setStreamingThoughts('');
                finalAnswerStarted = true;
                setShowThoughtsToggle(true);
              }
              setFinalResponse(prev => prev + data.char);
            } else if (data.type === 'error_stream' && typeof data.char === 'string') {
               setStreamingThoughts(prev => prev + data.char);
               setSavedThoughts(prev => prev + data.char);
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setFinalResponse('요청이 중단되었습니다.');
      } else {
        setFinalResponse(`⚠️ 오류: ${err.message}`);
      }
      if (streamingThoughts.length > 0) {
         setSavedThoughts(prev => prev + streamingThoughts);
      }
      setStreamingThoughts('');
      setShowThoughtsToggle(true);
      setIsThoughtsOpen(true);
    } finally {
      setIsLoading(false);
      setIsThinking(false);
      const end = performance.now();
      setElapsedTime(parseFloat(((end - start) / 1000).toFixed(1)));
      abortControllerRef.current = null;

      if (finalResponse || savedThoughts.length > 0) {
        setShowThoughtsToggle(true);
      }
    }
  };

  const handleCancel = () => { if (abortControllerRef.current) { abortControllerRef.current.abort(); } };
  const scrollToBottom = () => { window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' }); };

  const handleEdit = () => {
    setEditedContent(longTextContent);
    setIsEditing(true);
    setIsPollingPaused(true);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setIsPollingPaused(false);
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const response = await fetch('http://localhost:8000/write-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filepath: 'Frontend/public/longText.txt',
          content: editedContent,
        }),
      });
      if (!response.ok) {
        throw new Error('파일 저장에 실패했습니다.');
      }
      setLongTextContent(editedContent);
      setIsEditing(false);
    } catch (error) {
      console.error("저장 실패:", error);
      alert("파일 저장 중 오류가 발생했습니다.");
    } finally {
      setIsSaving(false);
      setIsPollingPaused(false);
    }
  };

  const isErrorContent = longTextContent === FILE_NOT_FOUND_MESSAGE;

  return (
    <div className={`min-h-screen bg-gray-100 flex flex-col p-6 font-[Pretendard] ${!isLongTextPaneOpen ? 'items-center' : ''}`}>
      <div className="w-full transition-all duration-300">
        <div className={`flex flex-row gap-4 items-start ${!isLongTextPaneOpen ? 'justify-center' : ''}`}>

          <div className="bg-white shadow-lg rounded-2xl py-6 px-8 relative flex flex-col transition-all duration-300 w-full max-w-3xl">
            <h1 className="text-3xl font-bold text-sky-500 mb-6 text-center">Kaede</h1>
            <InputSection
              input={input}
              setInput={setInput}
              handleSend={handleSend}
              handleCancel={handleCancel}
              isLoading={isLoading}
              textareaRef={textareaRef}
            />
            <hr className="my-6 border-gray-300" />
            <div>
              <h2 className="text-lg font-semibold text-gray-700 mb-2">
                카에데:
                {elapsedTime !== null && !isLoading && (<span className="text-sm text-gray-500 ml-2">⏱️ {elapsedTime}s</span>)}
              </h2>
              <div ref={responseContainerRef} className="p-4 rounded-xl prose prose-max-w-none bg-gray-100 text-gray-800 min-h-[100px] relative">
                {isLoading && isThinking && !streamingThoughts && (
                  <div className="absolute inset-0 flex flex-col justify-center items-center z-20 text-gray-500">
                    <Spinner />
                    <p className="mt-2">카에데가 답변을 생각중이에요!</p>
                  </div>
                )}
                {isLoading && streamingThoughts && (
                  <div className="absolute inset-0 flex justify-center items-center z-20">
                    <Spinner />
                  </div>
                )}
                {!isLoading && !finalResponse && !streamingThoughts && !isThinking && (
                  <p className="text-center text-gray-500">무엇을 도와드릴까요?</p>
                )}
                <div className="relative z-10">
                  {streamingThoughts && (
                    <div className="mb-1 text-gray-500 whitespace-pre-wrap">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamingThoughts}</ReactMarkdown>
                    </div>

                  )}
                  {finalResponse && (
                    <div className="mb-1 text-gray-800 mt-4">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{finalResponse}</ReactMarkdown>
                    </div>
                  )}
                  {showThoughtsToggle && (
                    <div className="mt-2 text-right">
                      <button
                        onClick={() => setIsThoughtsOpen(!isThoughtsOpen)}
                        className="text-sm text-indigo-600 hover:text-indigo-800 flex items-center justify-end"
                      >
                        {isThoughtsOpen ? '사고 과정 숨기기' : '사고 과정 보기'}
                        <svg className={`ml-1 w-4 h-4 transform transition-transform ${isThoughtsOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                        </svg>
                      </button>
                      <AnimatePresence>
                        {isThoughtsOpen && savedThoughts && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.3 }}
                            className="text-left p-3 mt-2 bg-gray-200 rounded-lg text-gray-600 overflow-hidden text-sm whitespace-pre-wrap"
                          >
                            <h4 className="font-semibold mb-1">Thought:</h4>
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{savedThoughts}</ReactMarkdown>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

            <AnimatePresence mode="wait">
              {isLongTextPaneOpen ? (
                <motion.div
                  key="long-text-panel"
                  className="flex-grow bg-white shadow-lg rounded-2xl p-8 flex flex-col min-w-0 max-w-full"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ duration: 0.3, ease: "easeInOut" }}
                  onAnimationComplete={adjustHeight}
                >
                  <div className="flex justify-between items-center mb-2">
                    <div className="flex items-center gap-4">
                      <button onClick={() => setIsLongTextPaneOpen(false)} className="text-sm font-medium text-indigo-600 hover:text-indigo-800">숨기기</button>
                    </div>
                  </div>
                  <div
                    ref={longTextContentRef}
                    key={renderMode}
                    className="bg-indigo-50 p-4 pb-15 rounded-xl text-gray-800 prose prose-max-w-none min-w-0 max-w-full aspect-video overflow-hidden"
                  >
                    <div className="flex justify-end items-center mb-2 -mt-2 -mr-2">
                      <div className="flex items-center space-x-4">
                        {renderMode === 'text' && !isEditing && !isErrorContent && (
                          <button onClick={handleEdit} className="text-sm font-medium text-indigo-600 hover:text-indigo-800">편집</button>
                        )}
                        {renderMode === 'text' && isEditing && (
                          <>
                            <button onClick={handleSave} disabled={isSaving} className="text-sm font-medium text-green-600 hover:text-green-800 disabled:opacity-50">
                              {isSaving ? '저장 중...' : '저장'}
                            </button>
                            <button onClick={handleCancelEdit} className="text-sm font-medium text-gray-600 hover:text-gray-800">취소</button>
                          </>
                        )}
                        <div className="flex items-center">
                          <span className={`text-sm font-medium mr-2 ${renderMode === 'text' ? 'text-gray-900' : 'text-gray-500'}`}>원본</span>
                          <button onClick={() => setRenderMode(renderMode === 'preview' ? 'text' : 'preview')} className={`relative inline-flex items-center h-6 rounded-full w-11 transition-colors ${renderMode === 'preview' ? 'bg-indigo-600' : 'bg-gray-200'}`}>
                            <span className={`inline-block w-4 h-4 transform bg-white rounded-full transition-transform ${renderMode === 'preview' ? 'translate-x-6' : 'translate-x-1'}`} />
                          </button>
                          <span className={`text-sm font-medium ml-2 ${renderMode === 'preview' ? 'text-gray-900' : 'text-gray-500'}`}>미리보기</span>
                        </div>
                      </div>
                    </div>
                    <LongTextViewer
                      renderMode={renderMode}
                      isHtmlContent={isHtmlContent}
                      longTextContent={longTextContent}
                      iframeRef={iframeRef}
                      longTextViewerRef={longTextViewerRef}
                      isEditing={isEditing}
                      editedContent={editedContent}
                      onContentChange={(e) => setEditedContent(e.target.value)}
                    />
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="open-panel-button"
                  className="flex-shrink-0"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3, ease: "easeInOut" }}
                >
                  <button onClick={() => setIsLongTextPaneOpen(true)} className="w-16 h-16 bg-indigo-500 text-white rounded-2xl flex items-center justify-center shadow-lg hover:bg-indigo-600 transition-all transform hover:scale-110">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

        </div>
      </div>

      <button ref={scrollToBottomButtonRef} onClick={scrollToBottom} className="fixed bottom-20 left-1/2 -translate-x-1/2 bg-indigo-500 text-white rounded-full w-12 h-12 flex items-center justify-center shadow-lg hover:bg-indigo-600 transition-opacity" style={{ display: 'none' }}>
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" /></svg>
      </button>
    </div>
  );
}

export default App;
