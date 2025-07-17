import { useRef, useState, useEffect } from 'react';
import './App.css';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// 스피너 컴포넌트 (이전과 동일)
const Spinner = () => (
  <div className="flex justify-center items-center h-full">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
  </div>
);

function App() {
  const [input, setInput] = useState('');
  const [displayedResponse, setDisplayedResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [elapsedTime, setElapsedTime] = useState<number | null>(null);
  const [cursorVisible, setCursorVisible] = useState(true);
  const currentTextRef = useRef('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const typingIntervalRef = useRef<number | null>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // ⏺️ 커서 깜빡임
  useEffect(() => {
    if (!isLoading) {
      setCursorVisible(true);
      return;
    }
    const timer = setInterval(() => setCursorVisible(prev => !prev), 500);
    return () => clearInterval(timer);
  }, [isLoading]);

  // ⬇️ 자동 스크롤
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [displayedResponse]);

  // textarea 자동 확장 로직
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.max(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [input]);

  const handleSend = async () => {
    setDisplayedResponse('');
    currentTextRef.current = '';
    setInput('');
    setElapsedTime(null);
    setIsLoading(true);
    if (typingIntervalRef.current) clearInterval(typingIntervalRef.current);

    const start = performance.now();

    try {
      const res = await fetch('http://localhost:8000/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: input }),
      });

      const data = await res.json();
      const fullText = data.response || data.error || '(no response)';

      const chars = Array.from(fullText);
      let i = 0;

      if (typingIntervalRef.current) clearInterval(typingIntervalRef.current);

      typingIntervalRef.current = window.setInterval(() => {
        if (i < chars.length) {
          currentTextRef.current += chars[i];
          setDisplayedResponse(currentTextRef.current);
          i++;
        } else {
          clearInterval(typingIntervalRef.current!);
          setIsLoading(false);
          const end = performance.now();
          setElapsedTime(parseFloat(((end - start) / 1000).toFixed(1)));
        }
      }, 30);
    } catch (err) {
      setDisplayedResponse(`⚠️ 오류: ${String(err)}`);
      setIsLoading(false);
      const end = performance.now();
      setElapsedTime(parseFloat(((end - start) / 1000).toFixed(1)));
    }
  };

  const handleCancel = () => {
    if (typingIntervalRef.current) clearInterval(typingIntervalRef.current);
    setIsLoading(false);
    setElapsedTime(null);
    setDisplayedResponse('요청이 중단되었습니다.');
    setCursorVisible(true);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-6 font-[Pretendard]">
      <div className="w-full max-w-3xl bg-white shadow-lg rounded-2xl p-8">
        <h1 className="text-3xl font-bold text-indigo-600 mb-6 text-center">Kaede</h1>

        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="카에데에게 말걸어주세요!"
          className="w-full p-4 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-transparent transition resize-none min-h-[120px] overflow-hidden"
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

        <hr className="my-6 border-gray-300" />

        <div>
          <h2 className="text-lg font-semibold text-gray-700 mb-2">
            카에데:
            {elapsedTime !== null && !isLoading && (
              <span className="text-sm text-gray-500 ml-2">⏱️ {elapsedTime}s</span>
            )}
          </h2>
          {/* ✨ 여기 클래스를 'markdown-body' 대신 'prose'로 변경합니다. */}
          {/* 또한, 텍스트가 시작될 때 스피너 자리에 붙어 나올 수 있도록 기본 justify-start로 설정합니다. */}
          {/* ✨ 여기 클래스에 'whitespace-normal'을 추가합니다. */}
          <div className="bg-gray-100 p-4 rounded-xl text-gray-800 min-h-[100px] prose whitespace-pre-wrap">
            {isLoading && displayedResponse === '' ? (
              <div className="flex justify-center items-center h-[inherit]">
                <Spinner />
              </div>
            ) : (
              <>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {displayedResponse}
                </ReactMarkdown>
                {isLoading && cursorVisible && (
                  <span className="inline-block align-bottom -translate-y-[2px]">|</span>
                )}
              </>
            )}
            <div ref={scrollRef}></div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;