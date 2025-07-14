import { useState } from 'react';
import { sendMessageToGemini } from './api';

function App() {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');

  const handleSend = async () => {
    const result = await sendMessageToGemini(input);
    setResponse(result);
  };

  return (
    <div style={{ padding: '2rem' }}>
      <h1>🧠 ToyGemini Chat</h1>
      <textarea
        rows={5}
        cols={60}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="명령어를 입력하세요"
      />
      <br />
      <button onClick={handleSend}>Gemini에게 보내기</button>
      <hr />
      <div>
        <strong>Gemini 응답:</strong>
        <pre>{response}</pre>
      </div>
    </div>
  );
}

export default App;
