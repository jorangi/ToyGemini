import axios from 'axios';

export const sendMessageToGemini = async (message: string) => {
  try {
    const res = await axios.post('http://127.0.0.1:8000/gemini/message', {
      message,
    });
    return res.data.response;
  } catch (err) {
    console.error('에러:', err);
    return '서버 에러';
  }
};
